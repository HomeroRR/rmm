''' Evaluation of agent trajectories '''

import json
import os
import sys
from collections import defaultdict
import networkx as nx
import numpy as np
import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent=4)

from env import R2RBatch
from utils import load_datasets, load_nav_graphs, dialog_to_string
from agent import BaseAgent, StopAgent, RandomAgent, ShortestAgent


class Evaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits, path_type='planner_path', datasets='NDH', mount_dir='', segmented=False, speaker_only=False, results_dir=None, steps_to_next_q=4):
        self.error_margin = 3.0
        self.splits = splits
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        self.steps_to_next_q = steps_to_next_q
        for item in load_datasets(splits, datasets=datasets, mount_dir=mount_dir, segmented=segmented, speaker_only=speaker_only):
            self.gt[item['inst_idx']] = item
            self.instr_ids.append(item['inst_idx'])
            self.scans.append(item['scan'])

            # Add 'trusted_path' to gt metadata if necessary.
            if path_type == 'trusted_path':
                planner_goal = item['planner_path'][-1]
                if planner_goal in item['player_path'][1:]:
                    self.gt[item['inst_idx']]['trusted_path'] = item['player_path'][:]
                else:
                    self.gt[item['inst_idx']]['trusted_path'] = item['planner_path'][:]
        print("number of segments is "+ str(len(self.gt)))
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        self.path_type = path_type
        for scan,G in self.graphs.iteritems(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

        ceiling = []
        self.gps = defaultdict(list)
        avg_num_steps = [0]
        for item in self.gt.values():
            start = item['player_path'][0]
            end = item['player_path'][-1]
            ceiling.append(self.distances[item['scan']][start][end])
            num_instrs = int(len(item['dialog_history'])/2)
            avg_num_steps.append(len(item[self.path_type])*1.0/num_instrs)
            for i,curr in enumerate(item['player_path']):
                if i in item['request_locations']:
                    d1 = self.distances[item['scan']][start][item['end_panos'][0]]
                    d2 = self.distances[item['scan']][curr][item['end_panos'][0]]
                    self.gps[item['inst_idx']].append(d1 - d2)
            self.gps[item['inst_idx']].append(self.distances[item['scan']][start][end])
        print("ceiling:")
        print(sum(ceiling)*1.0/len(ceiling))

        print("Average number of steps between questions")
        print(np.mean(avg_num_steps))
        if splits[0] == 'val_unseen' and results_dir!=None:
            max_len = -float('inf')
            for k in self.gps:
                if len(self.gps[k])>max_len:
                    max_len = len(self.gps[k])
            for k in self.gps:
                if len(self.gps[k])<max_len:
                    self.gps[k] += ['']*(max_len-len(self.gps[k]))
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
            pd.DataFrame(self.gps).T.to_csv(results_dir + '/human_unseen_gps.csv')

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path, num_instrs=0):
        ''' Calculate error based on the final position in trajectory, and also 
            the closest position (oracle stopping rule). '''
        gt = self.gt[instr_id]
        start = gt[self.path_type][0]
        if start != path[0][0]:
            print 'Result trajectories should include the start position' 
            return       
        distance = 0  # Work out the length of the path in meters
        prev = path[0]
        for i,curr in enumerate(path[1:]):
            if prev[0] != curr[0]:
                try:
                    self.graphs[gt['scan']][prev[0]][curr[0]]
                except KeyError as err:
                    print 'Error: The provided trajectory moves from %s to %s but the navigation graph contains no '\
                        'edge between these viewpoints. Please ensure the provided navigation trajectories '\
                        'are valid, so that trajectory length can be accurately calculated.' % (prev[0], curr[0])
                    return # ignore missing path transition in dataset
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
            if (i+1)%self.steps_to_next_q == 0:
                d1 = self.distances[gt['scan']][start][gt['end_panos'][0]]
                d2 = self.distances[gt['scan']][curr[0]][gt['end_panos'][0]]
                self.gps[instr_id].append(d1 - d2)
        goal = gt[self.path_type][-1]
        planner_goal = gt['planner_path'][-1]  # for calculating oracle planner success (e.g., passed over desc goal?)
        final_position = path[-1][0]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        nearest_planner_position = self._get_nearest(gt['scan'], planner_goal, path)
        dist_to_end_start = None
        dist_to_end_end = None
        for end_pano in gt['end_panos']:
            d = self.distances[gt['scan']][start][end_pano]
            if dist_to_end_start is None or d < dist_to_end_start:
                dist_to_end_start = d
            d = self.distances[gt['scan']][final_position][end_pano]
            if dist_to_end_end is None or d < dist_to_end_end:
                dist_to_end_end = d
        self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        self.scores['oracle_plan_errors'].append(self.distances[gt['scan']][nearest_planner_position][planner_goal])
        self.scores['ceiling'].append(dist_to_end_start)
        self.scores['dist_to_end_reductions'].append(dist_to_end_start - dist_to_end_end)
        self.scores['trajectory_lengths'].append(distance)
        self.scores['shortest_path_lengths'].append(self.distances[gt['scan']][start][goal])

        self.gps[instr_id].append(dist_to_end_start - dist_to_end_end)

    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        # self.gp_vs_qs = defaultdict(list)
        self.gps = defaultdict(list)
        instr_ids = set(self.instr_ids)
        with open(output_file) as f:
            for item in json.load(f):
                # Check against expected ids
                if item['inst_idx'] in instr_ids:
                    instr_ids.remove(item['inst_idx'])
                    self._score_item(item['inst_idx'], item['trajectory'], item['num_instrs'])
        assert len(instr_ids) == 0, 'Trajectories not provided for %d instruction ids: %s' % (len(instr_ids), instr_ids)
        # assert len(self.scores['nav_errors']) == len(self.instr_ids)

        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])
        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])
        oracle_plan_successes = len([i for i in self.scores['oracle_plan_errors'] if i < self.error_margin])

        spls = []
        for err, length, sp in zip(self.scores['nav_errors'], self.scores['trajectory_lengths'], self.scores['shortest_path_lengths']):
            if err < self.error_margin:
                if sp > 0:
                    spls.append(sp / max(length, sp))
                else:  # In IF, some Q/A pairs happen when we're already in the goal region, so taking no action is correct.
                    spls.append(1 if length == 0 else 0)
            else:
                spls.append(0)
        
        score_summary ={
            'length': np.average(self.scores['trajectory_lengths']),
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle success_rate': float(oracle_successes)/float(len(self.scores['oracle_errors'])),
            'success_rate': float(num_successes)/float(len(self.scores['nav_errors'])),
            'spl': np.average(spls),
            'oracle path_success_rate': float(oracle_plan_successes)/float(len(self.scores['oracle_plan_errors'])),
            'dist_to_end_reduction': sum(self.scores['dist_to_end_reductions']) / float(len(self.scores['dist_to_end_reductions'])),
            'ceiling': sum(self.scores['ceiling']) / float(len(self.scores['ceiling']))
        }

        max_len = -float('inf')
        for k in self.gps:
            if len(self.gps[k])>max_len:
                max_len = len(self.gps[k])
        for k in self.gps:
            if len(self.gps[k])<max_len:
                self.gps[k] += ['']*(max_len-len(self.gps[k]))
        assert score_summary['spl'] <= score_summary['success_rate']
        return score_summary, self.scores, self.gps

    def bleu_score(self, path2inst, for_nav=False, tok=None, use_dialog_history=False):
        from bleu import compute_bleu
        refs = []
        candidates = []
        for path_id, inst in path2inst.items():
            path_id = path_id
            assert path_id in self.gt
            # There are three references
            real_instructions, gen_ints = None, None
            if not use_dialog_history:
                instructions_type = 'ora_instructions'
                if for_nav:
                    instructions_type = 'nav_instructions'
                real_instructions = self.gt[path_id][instructions_type]
                gen_ints = [tok.index_to_word[word_id] for word_id in inst]
            else:
                real_instructions = dialog_to_string(self.gt[path_id]['dialog_history'])
                gen_ints = tok.split_sentence(dialog_to_string(inst))

            refs.append( [tok.split_sentence(real_instructions)] )
            candidates.append(gen_ints)

        tupl = compute_bleu(refs, candidates, smooth=True)
        bleu_s = tupl[0]
        precisions = tupl[1]

        return bleu_s, precisions



RESULT_DIR = 'tasks/NDH/results/'


def eval_simple_agents():
    # path_type = 'planner_path'
    # path_type = 'player_path'
    path_type = 'trusted_path'

    ''' Run simple baselines on each split. '''
    for split in ['train', 'val_seen', 'val_unseen']:
        env = R2RBatch(None, batch_size=1, splits=[split], path_type=path_type)
        ev = Evaluation([split], path_type=path_type)

        for agent_type in ['Stop', 'Shortest', 'Random']:
            outfile = '%s%s_%s_agent.json' % (RESULT_DIR, split, agent_type.lower())
            agent = BaseAgent.get_agent(agent_type)(env, outfile)
            agent.test()
            agent.write_results()
            score_summary, _ = ev.score(outfile)
            print '\n%s' % agent_type
            pp.pprint(score_summary)


def eval_seq2seq():
    ''' Eval sequence to sequence models on val splits (iteration selected from training error) '''
    outfiles = [
        RESULT_DIR + 'seq2seq_teacher_imagenet_%s_iter_5000.json',
        RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_20000.json'
    ]
    for outfile in outfiles:
        for split in ['val_seen', 'val_unseen']:
            ev = Evaluation([split])
            score_summary, _ = ev.score(outfile % split)
            print '\n%s' % outfile
            pp.pprint(score_summary)


if __name__ == '__main__':

    eval_simple_agents()
    #eval_seq2seq()
