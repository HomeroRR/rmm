''' Utils for io, language, connectivity graphs etc '''

import os
import sys
import re
import string
import json
import time
import math
import copy
import base64
import csv
import numpy as np
import networkx as nx
from collections import Counter, defaultdict

csv.field_size_limit(sys.maxsize)

import torch
import torch.distributions as D
import torch.nn.functional as F


# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>', '<NAV>', '<ORA>', '<TAR>']
padding_idx = base_vocab.index('<PAD>')

def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3], 
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def load_datasets(splits, datasets='NDH', mount_dir='', segmented=False, speaker_only=False):
    data = []
    for split in splits:
        assert split in ['train', 'val_seen', 'val_unseen', 'test']
        for dataset_type in datasets.split("_"):
            data_source = mount_dir
            if dataset_type == 'R2R':
                data_source += 'tasks/' + dataset_type + '/data/' + dataset_type + '_%s.json' 
            else:
                data_source += 'tasks/' + dataset_type + '/data/%s.json'
            print('Using ' + dataset_type + ' for '+ split +'!\n\n\n')
            with open( data_source % split) as f:
                items = json.load(f)
                new_items = []
                for item in items:
                    if dataset_type == 'R2R':
                        item['target'] = '<UNK>' 
                        item['inst_idx'] = item['path_id']
                        item['dialog_history'] = [
                            {"nav_idx": 0, "role": "navigator", "message": "I am lost . where should I go next ?"},
                            {"nav_idx": 0, "role": "oracle", "message": item['instructions'][0]}
                            ]
                        item['planner_path'] = item['path']
                        item['player_path'] = item['path']
                        item['nav_history'] = item['player_path']
                        item['start_pano'] = {'heading': item['heading'],'elevation': 17.5,'pano':item['path'][0]}
                    elif dataset_type == 'CVDN':
                        item['inst_idx'] = item['idx']
                        item['planner_path'] = item['planner_nav_steps']
                        item['player_path'] = item['nav_steps']
                        item['nav_history'] = item['player_path']
                        heading, elevation = 2.0, 17.5
                        if 'nav_camera' in item and len(item['nav_camera']) > 0:
                            nav_camera = item['nav_camera'][0]
                            if 'message' in nav_camera:
                                heading = nav_camera['message'][-1]['heading']
                                elevation = nav_camera['message'][-1]['elevation']
                        item['start_pano'] = {'heading': heading,'elevation': elevation,'pano':item['planner_nav_steps'][0]}
                    nav_ins, ora_ins, request_locations, nav_seen, ora_seen, nav_idx = [], [], {}, [], [], 0
                    for index, turn in enumerate(item['dialog_history']):
                        if turn['role'] == 'navigator':
                            nav_ins.append(turn['message'])
                            if len(ora_seen) > 0:
                                request_locations[nav_idx] = [' '.join(nav_seen), ' '.join(ora_seen), index]
                                ora_seen = []
                                nav_seen = []
                            nav_seen.append(turn['message'])
                        else:
                            ora_ins.append(turn['message'])
                            if len(nav_seen) > 0:
                                nav_idx = int(turn['nav_idx'])
                                ora_seen.append(turn['message'])
                    if len(ora_seen) > 0:
                        request_locations[nav_idx] = [nav_seen[-1], ora_seen[-1], len(item['dialog_history'])]  #[' '.join(nav_seen), ' '.join(ora_seen), len(item['dialog_history'])] 
                    item['nav_instructions'] = ' '.join(nav_ins)
                    item['ora_instructions'] = ' '.join(ora_ins)
                    if len(item['nav_instructions'])==0 or len(item['ora_instructions'])==0:
                        continue 
                    item['request_locations'] = request_locations
                    item['inst_idx'] = str(item['inst_idx'])
                    assert len(item['player_path']) >1, item['player_path'] 
                    new_items.append(item)
                if not segmented:
                    data += new_items
                else:
                    spread_out_items = []
                    for item in new_items:
                        # Split multiple instructions into separate entries
                        spread_counter = 0
                        turn_items = 1 
                        nav_histories = []
                        exchange_points = set()
                        first_nav_index = int(item['dialog_history'][0]['nav_idx'])
                        if not speaker_only:
                            if first_nav_index > 0:
                                spread_out_item = dict(item)
                                spread_out_item['inst_idx'] = item['inst_idx'] + '_' + str(spread_counter)
                                spread_out_item['dialog_history'] = [
                                    {"nav_idx": 0, "role": "navigator", "message": item['target']},
                                    {"nav_idx": 0, "role": "oracle", "message": item['target']}
                                    ]
                                spread_out_item['player_path'] = item['nav_history'][:first_nav_index+1]
                                spread_out_item['planner_path'] = spread_out_item['player_path']
                                spread_out_item['end_panos']  = [item['nav_history'][first_nav_index]]
                                spread_out_item['nav_instructions'] = item['target']
                                spread_out_item['ora_instructions'] = item['target']
                                spread_out_items.append(spread_out_item)
                                nav_histories.append(spread_out_item['player_path'])
                                spread_counter +=1
                                turn_items += len(spread_out_item['player_path'])-1
                        len_dialog_history=len(item['dialog_history'])
                        len_path = len(item['nav_history'])
                        for index, turn in enumerate(item['dialog_history'][:-1]):
                            nav_idx = int(turn['nav_idx'])
                            if nav_idx in exchange_points:
                                continue
                            if turn['role'] == 'navigator':
                                next_turn =  item['dialog_history'][index+1]
                                if next_turn['role'] != 'navigator':
                                    for j in range(index+1, len_dialog_history):
                                        next_next_turn =  item['dialog_history'][j]
                                        if next_next_turn['role'] == 'navigator' or j+1 == len_dialog_history:
                                            spread_out_item = dict(item)
                                            end_idx = int(next_next_turn['nav_idx'])+1 if j+1 < len_dialog_history else len_path #To next NAV or goal
                                            if end_idx - nav_idx < 2 or nav_idx >= len_path or end_idx >  len_path: # path must be valid
                                                continue
                                            spread_out_item['inst_idx'] = item['inst_idx'] + '_' + str(spread_counter)# to string
                                            spread_out_item['dialog_history'] = [
                                                {"nav_idx": nav_idx, "role": "navigator", "message": turn['message']},
                                                {"nav_idx": nav_idx, "role": "oracle", "message": next_turn['message']}
                                                ]
                                            spread_out_item['player_path'] = item['nav_history'][nav_idx:end_idx]
                                            spread_out_item['planner_path'] = spread_out_item['player_path']
                                            # spread_out_item['start_pano'] = {'heading': item['start_pano']['heading'],'elevation': 17.5,'pano':spread_out_item['player_path'][0]}
                                            spread_out_item['end_panos']  = [item['nav_history'][end_idx-1]]
                                            spread_out_item['nav_instructions'] = turn['message']
                                            spread_out_item['ora_instructions'] = next_turn['message']
                                            
                                            spread_out_items.append(spread_out_item)
                                            nav_histories.append(spread_out_item['player_path'])
                                            spread_counter +=1
                                            exchange_points.add(nav_idx)
                                            turn_items += len(spread_out_item['player_path']) - 1
                                            break
                        # assert turn_items == len(item['nav_history']) or len(item['nav_history']) < 2 ,  str(turn_items) + ' ' + str(len(item['nav_history']))
                    data += spread_out_items
                                
    return data


class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')  # Split on any non-alphanumeric character
  
    def __init__(self, vocab=None, encoding_length=20):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        self.index_to_word = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i
            new_w2i = defaultdict(lambda: self.word_to_index['<UNK>'])
            new_w2i.update(self.word_to_index)
            self.word_to_index = new_w2i
            for key, value in self.word_to_index.items():
                self.index_to_word[value] = key
        self.add_word('<BOS>')
        print("VOCAB_SIZE", self.vocab_size())

    def finalize(self):
        """
        This is used for debug
        """
        self.word_to_index = dict(self.word_to_index)   # To avoid using mis-typing tokens

    def add_word(self, word):
        assert word not in self.word_to_index
        self.word_to_index[word] = self.vocab_size()    # vocab_size() is the
        self.index_to_word[self.vocab_size()] = word
    
    def split_sentence(self, sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def vocab_size(self):
        return len(self.index_to_word)

    def encode_sentence(self, sentences, seps=None):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []
        if type(sentences) is not list:
            sentences = [sentences]
            seps = [seps]
        for sentence, sep in zip(sentences, seps):
            if sep is not None:
                encoding.append(self.word_to_index[sep])
            for word in self.split_sentence(sentence)[::-1]:  # reverse input sentences
                if word in self.word_to_index:
                    encoding.append(self.word_to_index[word])
                else:
                    encoding.append(self.word_to_index['<UNK>'])
        encoding.append(self.word_to_index['<EOS>'])
        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length-len(encoding))

        # cut off the LHS of the encoding if it's over-size (e.g., words from the end of an individual command,
        # favoring those at the beginning of the command (since inst word order is reversed) (e.g., cut off the early
        # instructions in a dialog if the dialog is over size, preserving the latest QA pairs).
        prefix_cut = max(0, len(encoding) - self.encoding_length)
        return np.array(encoding[prefix_cut:])

    def decode_sentence(self, encoding):
        sentence = []
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                try:
                    word = self.index_to_word[ix]
                    sentence.append(word)
                except:
                    pass
                    # print("Missing index %d" % ix )
        return " ".join(sentence[::-1]) # unreverse before output

    def shrink(self, inst):
        """
        :param inst:    The id inst
        :return:  Remove the potential <BOS> and <EOS>
                  If no <EOS> return empty list
        """
        if len(inst) == 0:
            return inst
        end = np.argmax(np.array(inst) == self.word_to_index['<EOS>'])     # If no <EOS>, return empty string
        if len(inst) > 1 and inst[0] == self.word_to_index['<BOS>']:
            start = 1
        else:
            start = 0
        # print(inst, start, end)
        return inst[start: end]

def build_vocab(splits=['train'], min_count=5, start_vocab=base_vocab):
    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    count = Counter()
    t = Tokenizer()
    data = load_datasets(splits)
    for item in data:
        for turn in item['dialog_history']:
            count.update(t.split_sentence(turn['message']))
    vocab = list(start_vocab)

    # Add words that are object targets.
    targets = set()
    for item in data:
        target = item['target']
        targets.add(target)
    vocab.extend(list(targets))

    # Add words above min_count threshold.
    for word, num in count.most_common():
        if word in vocab:  # targets strings may also appear as regular vocabulary.
            continue
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab


def write_vocab(vocab, path):
    print 'Writing vocab of size %d to %s' % (len(vocab),path)
    with open(path, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)


def read_vocab(path):
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def get_optimizer_constructor(optim='rms'):
    optimizer = None
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif optim == 'adamax':
        print("Optimizer: adamax")
        optimizer = torch.optim.Adamax
    else:
        assert False
    return optimizer

def load_features(feature_store, blind):
    features = None
    image_w = 640
    image_h = 480
    vfov = 60 
    feature_size = 0
    if feature_store:
        print 'Loading image features from %s' % feature_store
        if blind:
            print("... and zeroing them out for 'blind' evaluation")
        tsv_fieldnames = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
        features = {}
        with open(feature_store, "r+b") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = tsv_fieldnames)
            for item in reader:
                image_h = int(item['image_h'])
                image_w = int(item['image_w'])
                vfov = int(item['vfov'])
                long_id = item['scanId'] + '_' + item['viewpointId']
                if not blind:
                    features[long_id] = np.frombuffer(base64.decodestring(item['features']),
                            dtype=np.float32).reshape((36, 2048))
                else:
                    features[long_id] = np.zeros((36, 2048), dtype=np.float32)
        feature_size = next(iter(features.values())).shape[-1]
        print('The feature size is %d' % feature_size)
    else:
        print 'Image features not provided'
    return features, {'image_w':image_w, 'image_h':image_h, 'vfov':vfov, 'feature_size':feature_size}

''' 
    The following code was extracted from sota_cvdn
    -----------------------------------------------
    -----------------------------------------------
    -----------------------------------------------
    -----------------------------------------------
    -----------------------------------------------
'''

def angle_feature(heading, elevation, angle_feat_size=4):
    return np.array([math.sin(heading), math.cos(heading),
                     math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
                    dtype=np.float32)

def get_point_angle_feature(baseViewId=0, angle_feat_size=4):
    sim = new_simulator()

    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            sim.newEpisode('ZMojNkEp431', '2f4d90acd4024c269fb0efe49a8ac540', 0, math.radians(-30))
        elif ix % 12 == 0:
            sim.makeAction(0, 1.0, 1.0)
        else:
            sim.makeAction(0, 1.0, 0)

        state = sim.getState()
        assert state.viewIndex == ix

        heading = state.heading - base_heading

        feature[ix, :] = angle_feature(heading, state.elevation)
    return feature

def get_all_point_angle_feature():
    return [get_point_angle_feature(baseViewId) for baseViewId in range(36)]


def add_idx(inst):
    toks = Tokenizer.split_sentence(inst)
    return " ".join([str(idx)+tok for idx, tok in enumerate(toks)])

import signal
class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True

from collections import OrderedDict

class Timer:
    def __init__(self):
        self.cul = OrderedDict()
        self.start = {}
        self.iter = 0

    def reset(self):
        self.cul = OrderedDict()
        self.start = {}
        self.iter = 0

    def tic(self, key):
        self.start[key] = time.time()

    def toc(self, key):
        delta = time.time() - self.start[key]
        if key not in self.cul:
            self.cul[key] = delta
        else:
            self.cul[key] += delta

    def step(self):
        self.iter += 1

    def show(self):
        total = sum(self.cul.values())
        for key in self.cul:
            print("%s, total time %0.2f, avg time %0.2f, part of %0.2f" %
                  (key, self.cul[key], self.cul[key]*1./self.iter, self.cul[key]*1./total))
        print(total / self.iter)


stop_word_list = [
    ",", ".", "and", "?", "!"
]


def stop_words_location(inst, mask=False):
    toks = Tokenizer.split_sentence(inst)
    sws = [i for i, tok in enumerate(toks) if tok in stop_word_list]        # The index of the stop words
    if len(sws) == 0 or sws[-1] != (len(toks)-1):     # Add the index of the last token
        sws.append(len(toks)-1)
    sws = [x for x, y in zip(sws[:-1], sws[1:]) if x+1 != y] + [sws[-1]]    # Filter the adjacent stop word
    sws_mask = np.ones(len(toks), np.int32)         # Create the mask
    sws_mask[sws] = 0
    return sws_mask if mask else sws

def get_segments(inst, mask=False):
    toks = Tokenizer.split_sentence(inst)
    sws = [i for i, tok in enumerate(toks) if tok in stop_word_list]        # The index of the stop words
    sws = [-1] + sws + [len(toks)]      # Add the <start> and <end> positions
    segments = [toks[sws[i]+1:sws[i+1]] for i in range(len(sws)-1)]       # Slice the segments from the tokens
    segments = list(filter(lambda x: len(x)>0, segments))     # remove the consecutive stop words
    return segments

def clever_pad_sequence(sequences, batch_first=True, padding_value=0):
    max_size = sequences[0].size()
    max_len, trailing_dims = max_size[0], max_size[1:]
    max_len = max(seq.size()[0] for seq in sequences)
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
    if padding_value is not None:
        out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor

import torch
def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).cuda()
    return mask

def average_length(path2inst):
    length = []

    for name in path2inst:
        datum = path2inst[name]
        length.append(len(datum))
    return sum(length) / len(length)

def viewpoint_drop_mask(viewpoint, seed=None, drop_func=None):
    local_seed = hash(viewpoint) ^ seed
    torch.random.manual_seed(local_seed)
    drop_mask = drop_func(torch.ones(2048).cuda())
    return drop_mask

def save(best_model, epoch, iteration, path, with_critic=False):
    ''' Snapshot models '''
    the_dir, _ = os.path.split(path)
    if not os.path.isdir(the_dir):
        os.makedirs(the_dir)
    states = {}
    def create_state(name, model, optimizer):
        states[name] = {
            'epoch': epoch + 1,
            'iteration': iteration,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    all_tuple = [("encoder", best_model['encoder'], best_model['encoder_optm']),
                    ("decoder", best_model['decoder'], best_model['decoder_optm'])]
    if with_critic:
        all_tuple.append(("critic", best_model['critic'], best_model['critic_optm']))
    for param in all_tuple:
        create_state(*param)
    torch.save(states, path)

def load(best_model, path, with_critic=False, parallel=True):
    ''' Loads parameters (but not training state) '''
    states = torch.load(path)
    def recover_state(name, model, optimizer):
        state = model.state_dict()
        state.update(states[name]['state_dict'])
        # if parallel:
        #     new_state = {}
        #     for k, v in state.items():
        #         key = k[7:] if k[:7] == 'module.' else k# remove `module.`
        #         # print(key)
        #         new_state[key] = v
        #     state = new_state
        model.load_state_dict(state)
        optimizer.load_state_dict(states[name]['optimizer'])
    all_tuple = [("encoder", best_model['encoder'], best_model['encoder_optm']),
                    ("decoder", best_model['decoder'], best_model['decoder_optm'])]
    if with_critic:
        all_tuple.append(("critic", best_model['critic'], best_model['critic_optm']))
    for param in all_tuple:
        recover_state(*param)
    return states['encoder']['iteration']

# Determine next model inputs
def next_decoder_input(logit, feedback, temperature=None, all_env_action=[], batch_size=100, target=None):
    a_t = None
    if 'temperature' in feedback or 'penalty' in feedback:
        logit = logit * 1.0/temperature 
    if 'penalty' in feedback and len(all_env_action) > 0:
        taken_actions = {}
        for turn in all_env_action:
            for i in range(batch_size):
                if i not in taken_actions:
                    taken_actions[i] = set()
                taken_actions[i].add(turn[i])
        for i in range(batch_size):
            for v in taken_actions[i]:
                logit[i,v] *= temperature
    if feedback == 'teacher': 
        a_t = target                # teacher forcing
    elif feedback == 'argmax': 
        _,a_t = logit.max(1)        # student forcing - argmax
        a_t = a_t.detach()
    elif feedback == 'sample' or feedback == 'temperature' or feedback == 'penalty':
        probs = F.softmax(logit, dim=1)
        m = D.Categorical(probs)
        a_t = m.sample()            # sampling an action from model
    elif feedback == 'topk':
        k=3
        topk, sorted_indices = torch.topk(logit, k, dim=1)
        probs = F.softmax(topk, dim=1)
        m = D.Categorical(probs)
        s = m.sample()  
        a_t = sorted_indices.gather(1, s.unsqueeze(1)).squeeze()         
    elif 'nucleus' in feedback:
        p = 0.4
        coin = torch.ones(batch_size).float()*p
        b = D.Bernoulli(coin)
        flip = b.sample().int().cuda()
        u = D.Uniform(torch.zeros(batch_size),torch.ones(batch_size)*logit.size()[1])
        uniform = u.sample().int().cuda()  
        probs = F.softmax(logit, dim=1)
        m = D.Categorical(probs)
        categorical = m.sample().int()
        stack = torch.stack([uniform, categorical],1)
        a_t = stack.gather(1, flip.unsqueeze(1).long()).squeeze()
    else:
        sys.exit('Invalid feedback option')
    return a_t

def dialog_to_string(dialog):
    dia_inst = ''
    sentences = []
    seps = []
    for turn in dialog:
        sentences.append(turn['message'])
    return ' '.join(sentences)

def copy_dialog_history(obs):
    new_obs = []
    for ob in obs:
        new_obs.append({
            'inst_idx': ob['inst_idx'],
            'scan': ob['scan'],
            'viewpoint': ob['viewpoint'],
            'viewIndex': ob['viewIndex'],
            'heading': ob['heading'],
            'elevation': ob['elevation'],
            'feature': ob['feature'],
            'candidate': ob['candidate'],
            'step': ob['step'],
            'navigableLocations': ob['navigableLocations'],
            'instructions': ob['instructions'],
            'teacher': ob['teacher'],
            'generated_dialog_history': copy.deepcopy(ob['generated_dialog_history']),
            'instr_encoding': ob['instr_encoding'],
            'nav_instr_encoding': ob['nav_instr_encoding'],
            'ora_instr_encoding': ob['ora_instr_encoding'],
            'distance': ob['distance'],
            'action_probs': ob['action_probs']
        })
    return new_obs