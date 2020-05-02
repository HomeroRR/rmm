import torch
import torch.nn as nn
import numpy as np
import os
import copy
import utils
import model
import torch.nn.functional as F


class Speaker():
    env_actions = {
        'left': (0,-1, 0), # left
        'right': (0, 1, 0), # right
        'up': (0, 0, 1), # up
        'down': (0, 0,-1), # down
        'forward': (1, 0, 0), # forward
        '<end>': (0, 0, 0), # <end>
        '<start>': (0, 0, 0), # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(
        self, env, agent, rnn_dim=512, wemb=256, dropout=0.5, bidir=True, lr=0.0001, optim ='rms', 
        parallelize=True, temperature=0.6, feedback='argmax', maxDecode=120, speaker_rl=False ):
        self.env = env
        self.feature_size = self.env.env.feature_size
        self.env.tok.finalize()
        self.agent = agent
        self.helper_agent = None
        self.asker_oracle = None
        self.temperature = temperature
        self.feedback = feedback
        self.maxDecode = maxDecode
        self.rnn_dim = rnn_dim

        self.speaker_rl = speaker_rl

        # Model
        print("VOCAB_SIZE", self.env.tok.vocab_size())
        self.encoder = model.SpeakerEncoder(self.feature_size, rnn_dim, dropout, bidirectional=bidir).cuda()
        if parallelize:
            self.encoder = nn.DataParallel(self.encoder).cuda()
        self.decoder = model.SpeakerDecoder(self.env.tok.vocab_size(), wemb, self.env.tok.word_to_index['<PAD>'],
                                            rnn_dim, dropout).cuda()
        if parallelize:
            self.decoder = nn.DataParallel(self.decoder).cuda()
        optimizer = utils.get_optimizer_constructor(optim=optim)
        self.encoder_optimizer = optimizer(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optimizer(self.decoder.parameters(), lr=lr)

        # Evaluation
        self.softmax_loss = torch.nn.CrossEntropyLoss(ignore_index=self.env.tok.word_to_index['<PAD>'], reduction='none')

        # Will be used in beam search
        self.nonreduced_softmax_loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.env.tok.word_to_index['<PAD>'],
            size_average=False,
            reduce=False
        )

    def train(self, iters, do_reset=True, train_nav=False, return_predict=False, k=1, 
        current_t=0, prev_act=None, prev_a_t=None, prev_traj=None, prev_loss=0, prev_ended=None, prev_obs=None, 
        prev_dialog=None, perm_idx=None, help_requesters=None, prev_last_dist=None, prev_rewards=None, prev_hidden_states=None, 
        prev_policy_log_probs=None, prev_masks=None, prev_entropys=None, prev_ml_loss=None, train_rl=False):
        for i in range(iters):
            if do_reset:
                self.env.reset()

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            follower_distance, loss, agent_loss, rl_loss, traj, new_obs = self.teacher_forcing(train=True, for_nav=train_nav, 
                k=k, current_t=current_t, prev_act=prev_act, prev_a_t=prev_a_t, prev_traj=prev_traj, 
                prev_loss=prev_loss, prev_ended=prev_ended, prev_obs=prev_obs, prev_dialog=prev_dialog,
                perm_idx=perm_idx, help_requesters=help_requesters, prev_last_dist=prev_last_dist, 
                prev_rewards=prev_rewards, prev_hidden_states=prev_hidden_states, prev_policy_log_probs=prev_policy_log_probs, 
                prev_masks=prev_masks, prev_entropys=prev_entropys, prev_ml_loss=prev_ml_loss, train_rl=train_rl)

            loss.backward()
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            if return_predict:
                return follower_distance, agent_loss, rl_loss, traj, new_obs

    def get_insts(self, for_nav=False, wrapper=(lambda x: x)):
        # Get the caption for all the data
        self.env.reset_epoch(shuffle=True)
        path2inst = {}
        total = self.env.size()
        for _ in wrapper(range(total // self.env.batch_size + 1)):  # Guarantee that all the data are processed
            obs = self.env.reset()
            insts, _ = self.infer_batch(for_nav=for_nav)  # Get the insts of the result
            path_ids = [ob['inst_idx'] for ob in obs]  # Gather the path ids
            for path_id, inst in zip(path_ids, insts):
                if path_id not in path2inst:
                    path2inst[path_id] = self.env.tok.shrink(inst)  # Shrink the words
        return path2inst

    def valid(self, for_nav=False,*aargs, **kwargs):
        """

        :param iters:
        :return: path2inst: path_id --> inst (the number from <bos> to <eos>)
                 loss: The XE loss
                 word_accu: per word accuracy
                 sent_accu: per sent accuracy
        """
        path2inst = self.get_insts(for_nav=for_nav,*aargs, **kwargs)

        # Calculate the teacher-forcing metrics
        self.env.reset_epoch(shuffle=True)
        N = 1 
        metrics = np.zeros(3)
        for i in range(N):
            self.env.reset()
            metrics += np.array(self.teacher_forcing(train=False, for_nav=for_nav))
        metrics /= N

        return [path2inst] + [metric for metric in metrics]


    def _candidate_variable(self, obs, actions):
        candidate_feat = np.zeros((len(obs), self.feature_size), dtype=np.float32)
        for i, (ob, act) in enumerate(zip(obs, actions)):
            if act == self.agent.model_actions.index('<end>') or act == self.agent.model_actions.index('<ignore>'):  # Ignore or Stop --> Just use zero vector as the feature
                pass
            else:
                candidate_feat[i, :] = ob['candidate'][act]['feature'] # Image feat
        return torch.from_numpy(candidate_feat).cuda()
    
    
    def from_shortest_path(self, for_nav=False):
        """
        :return:
        """
        obs = self.env._get_obs()
        ended = np.array([False] * len(obs)) # Indices match permuation of the model, not env
        length = np.zeros(len(obs), np.int64)
        img_feats = []
        can_feats = []
        count = 0
        count_limit = 1 if for_nav else 5 #self.agent.episode_len
        env_actions_instructions = []
        while not ended.all() and count<count_limit:
            img_feats.append(self.agent._feature_variable(obs))
            teacher_action = self.agent._teacher_action(obs, ended)
            teacher_action = teacher_action.cpu().numpy()
            can_feats.append(self._candidate_variable(obs, teacher_action))
            self.env.step([self.agent.env_actions[action] for action in teacher_action])
            ints = np.array([self.agent.env_actions_instructions[action] for action in teacher_action])
            env_actions_instructions.append(torch.from_numpy(ints).cuda().view(-1,1))
            length += (1 - ended)
            ended[:] = np.logical_or(ended, (teacher_action == self.agent.model_actions.index('<end>')))
            obs = self.env._get_obs()
            count +=1
        img_feats = torch.stack(img_feats, 1).contiguous()  # batch_size, max_len, 36, 2048
        can_feats = torch.stack(can_feats, 1).contiguous()  # batch_size, max_len, 2048
        return (img_feats, can_feats), length, torch.stack(env_actions_instructions,1).squeeze(2)

    def gt_words(self, obs, for_nav=False):
        """
        See "utils.Tokenizer.encode_sentence(...)" for "instr_encoding" details
        """
        encoding_type = 'ora_instr_encoding'
        if for_nav:
            encoding_type = 'nav_instr_encoding'
        seq_tensor = np.array([ob[encoding_type] for ob in obs])
        return torch.from_numpy(seq_tensor).cuda()

    def teacher_forcing(
        self, train=True, features=None, insts=None, for_listener=False, for_nav=False, 
        extract_distance=False, eval=True, k=3, current_t=0, prev_act=None, prev_a_t=None,
        prev_traj=None, prev_loss=0, prev_ended=None, prev_obs=None, prev_dialog=None, perm_idx=None,
        help_requesters=None, prev_last_dist=None, prev_rewards=None, prev_hidden_states=None, 
        prev_policy_log_probs=None, prev_masks=None, prev_entropys=None, prev_ml_loss=None, train_rl=False):
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        # Get Image Input & Encode
        obs = self.env._get_obs()
        if features is not None:
            # It is used in calulating the speaker score in beam-search
            assert insts is not None
            (img_feats, can_feats), lengths = features
            ctx = self.encoder(can_feats, img_feats, lengths)
            batch_size = len(lengths)
        else:
            batch_size = len(obs)
            if not for_nav:
                if prev_act is not None:
                    self.env.reset(next_minibatch=False)
                    for env_act in prev_act:
                        self.env.step(env_act)
            (img_feats, can_feats), lengths, _ = self.from_shortest_path(for_nav=for_nav)      # Image Feature (from the shortest path)
            ctx = self.encoder(can_feats, img_feats, lengths)
        h_t = torch.zeros(1, batch_size, self.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, self.rnn_dim).cuda()
        ctx_mask = utils.length2mask(lengths)

        # Get Language Input
        if insts is None:
            insts = self.gt_words(obs,for_nav=for_nav)  # Language Feature

        # Decode
        logits, _, _ = self.decoder(insts, ctx, ctx_mask, h_t, c_t)

        # Because the softmax_loss only allow dim-1 to be logit,
        # So permute the output (batch_size, length, logit) --> (batch_size, logit, length)
        logits = logits.permute(0, 2, 1).contiguous()
        loss = self.softmax_loss(
            input  = logits[:, :, :-1],         # -1 for aligning
            target = insts[:, 1:]               # "1:" to ignore the word <BOS>
        )

        predict = np.zeros((len(obs), insts.size()[1]-1)) # Init with <PAD> index 0
        ora, follower_distance, agent_loss, rl_loss, traj, gen_dialog, speaker_loss = None, None, None, None, None, None, None
        if (self.helper_agent is not None and not for_nav) or (self.asker_oracle is not None and for_nav): #FRASE
            env = self.helper_agent.env if (self.helper_agent is not None and not for_nav) else self.asker_oracle.env
            follower_distances, cands, agent_losses, rl_losses, agent_trajs, gen_dialogs, speaker_losses = [], [], [], [], [], [], []
            for k_i in range(k):
                # nav = "for nav" if for_nav else ""
                # print("%d K_i: %d %s" % (k, k_i, nav) )
                cand, _ = self.infer_batch(sampling=False, train=train, featdropmask=None, img_feats=img_feats, can_feats=can_feats, 
                    lengths=lengths, insts=insts[:, 1:], feedback=self.feedback, for_nav=for_nav)
                cands.append(cand)
                
                new_obs = utils.copy_dialog_history(prev_obs)
                for i in range(batch_size):
                    if not (i in help_requesters):
                        continue
                    # instruction = env.tok.decode_sentence(cand[i])
                    # dia_inst = ''
                    # sentences = []
                    # seps = []
                    # sentences.append(instruction)
                    # sep = '<ORA>' if not for_nav else '<NAV>'
                    # seps.append(sep)
                    # dia_inst += sep + ' ' + instruction + ' '
                    # sentences.append(env.batch[i]['target'])
                    # seps.append('<TAR>')
                    # dia_inst += '<TAR> ' + env.batch[i]['target']
                    # env.batch[i]['instructions'] = dia_inst
                    # dia_enc = self.env.tok.encode_sentence(sentences, seps=seps)
                    # env.batch[i]['instr_encoding'] = dia_enc

                    instruction = self.env.tok.decode_sentence( self.env.tok.shrink(cand[perm_idx[i]]) )
                    role = 'navigator' if for_nav else 'oracle'
                    new_obs[i]['generated_dialog_history'].append({'role':role, 'message': instruction})

                if self.asker_oracle is not None and for_nav:
                    follower_distance, new_loss, agent_loss, rl_loss, traj, gen_dialog = self.asker_oracle.teacher_forcing(train=train,
                        eval=False, k=k, current_t=current_t, prev_act=prev_act, prev_a_t=prev_a_t, prev_traj=prev_traj, prev_loss=prev_loss,
                        prev_ended=prev_ended, prev_obs=new_obs, prev_dialog=prev_dialog, perm_idx=perm_idx, help_requesters=help_requesters,
                        prev_last_dist=prev_last_dist, prev_rewards=prev_rewards, prev_hidden_states=prev_hidden_states, prev_policy_log_probs=prev_policy_log_probs, 
                        prev_masks=prev_masks, prev_entropys=prev_entropys, prev_ml_loss=prev_ml_loss, train_rl=train_rl)
                    speaker_losses.append(new_loss)
                else:
                    follower_distance, agent_loss, rl_loss, traj, gen_dialog = self.helper_agent.rollout(reset=False, extract_distance=True, train=train,
                        speaker_branching=1, start_t=current_t, prev_all_env_action=prev_act, prev_a_t=prev_a_t, prev_traj=prev_traj, 
                        prev_loss=prev_loss, help_already_given=True, prev_ended=prev_ended, prev_obs=new_obs, prev_dialog=prev_dialog, used_perm_idx=perm_idx,
                        prev_last_dist=prev_last_dist, prev_rewards=prev_rewards, prev_hidden_states=prev_hidden_states, prev_policy_log_probs=prev_policy_log_probs, 
                        prev_masks=prev_masks, prev_entropys=prev_entropys, prev_ml_loss=prev_ml_loss, train_rl=train_rl)
                # print follower_distance
                follower_distances.append(follower_distance); agent_losses.append(agent_loss); agent_trajs.append(traj); gen_dialogs.append(gen_dialog); rl_losses.append(rl_loss)
                env.reset(next_minibatch=False)
            follower_distances = np.stack(follower_distances,1)
            # print follower_distances
            min_index = np.argmin(follower_distances, axis=1).flatten()#.index(min(follower_distances))
            min_index_rl = np.argmin(rl_losses)
            # print min_index
            follower_distance =  [ float("inf") for x in range(len(obs))] #torch.index_select(follower_distances, dim=1, index=min_index).squeeze()
            agent_loss, traj, gen_dialog = torch.zeros(len(obs)), [{} for x in range(len(obs))], [{} for x in range(len(obs))]
            for i,idx in enumerate(min_index):
                cand = cands[idx][i,:-1]
                predict[i,:cand.shape[0]] = cand
                follower_distance[i] = follower_distances[i,idx]
                agent_loss[i] = agent_losses[idx][i] 
                traj[i] = agent_trajs[idx][i]
                gen_dialog[i] = gen_dialogs[idx][i]
                if for_nav:
                    loss[i] += speaker_losses[idx][i]
            follower_distance = np.array(follower_distance).reshape((-1,1))
            rl_loss = rl_losses[min_index_rl]
            if self.speaker_rl:
                loss += rl_loss
            if for_nav:
                loss = loss.float().mean()
            torch.cuda.empty_cache()
            # speaker_type = "navigator" if for_nav else "oracle"
            # print("---------- %s --------" % speaker_type)
            # print("Regular loss: %0.4f " % loss.item())
            # loss += follower_distances[min_index] * 1.0/len(obs)
            # print("Follower loss: %0.4f " % follower_losses[min_index])
            # print "Min follower distances:"
            # print min_follower_distances
            # if k >1 and for_nav:
            #     print("Resolved at %d" % current_t)
        else:
            loss = loss.float().mean()
            if self.feedback != "teacher": #For baseline sampling
                _, loss2 = self.infer_batch(sampling=False, train=True, featdropmask=None, img_feats=img_feats, can_feats=can_feats, 
                    lengths=lengths, insts=insts[:, 1:], feedback=self.feedback, for_nav=for_nav)
                loss2 = loss2.float().mean()
                # print( "first loss: %0.4f" % loss)
                # print( "second loss: %0.4f" % (loss2))    
                loss = loss + loss2/100.0
                
        if for_listener:
            return self.nonreduced_softmax_loss(
                input  = logits[:, :, :-1],         # -1 for aligning
                target = insts[:, 1:]               # "1:" to ignore the word <BOS>
            )

        if extract_distance:
            return follower_distance, agent_loss, rl_loss, traj, gen_dialog
        elif train:
            return follower_distance, loss, agent_loss, rl_loss, traj, gen_dialog
        elif eval:
            # Evaluation  
            if (self.helper_agent is None or for_nav) and (self.asker_oracle is None or not for_nav): 
                _, predict = logits.max(dim=1)  # BATCH, LENGTH
            gt_mask = (insts != self.env.tok.word_to_index['<PAD>'])
            correct = (predict[:, :-1] == insts[:, 1:]) * gt_mask[:, 1:]    # Not pad and equal to gt
            correct, gt_mask = correct.type(torch.LongTensor), gt_mask.type(torch.LongTensor)
            word_accu = correct.sum().item() / gt_mask[:, 1:].sum().item()     # Exclude <BOS>
            sent_accu = (correct.sum(dim=1) == gt_mask[:, 1:].sum(dim=1)).sum().item() / batch_size  # Exclude <BOS>
            return loss.float().mean().item(), word_accu, sent_accu
        else:
            return follower_distance, loss, agent_loss, rl_loss, traj, gen_dialog


    def infer_batch(
        self, sampling=False, train=False, featdropmask=None, 
        img_feats=None, can_feats=None, lengths=None, insts=None, feedback='argmax',
        for_nav=False):
        """

        :param sampling: if not, use argmax. else use softmax_multinomial
        :param train: Whether in the train mode
        :return: if sampling: return insts(np, [batch, max_len]),
                                     log_probs(torch, requires_grad, [batch,max_len])
                                     hiddens(torch, requires_grad, [batch, max_len, dim})
                      And if train: the log_probs and hiddens are detached
                 if not sampling: returns insts(np, [batch, max_len])
        """
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        loss = 0

        # Image Input for the Encoder
        obs = self.env._get_obs()
        batch_size = len(obs)

        # Get feature
        if img_feats is None:
            (img_feats, can_feats), lengths, _ = self.from_shortest_path(for_nav=for_nav)      # Image Feature (from the shortest path)

        # This code block is only used for the featdrop.
        if featdropmask is not None:
            img_feats[...] *= featdropmask
            can_feats[...] *= featdropmask

        # Encoder
        ctx = self.encoder(can_feats, img_feats, lengths,
                           already_dropfeat=(featdropmask is not None))
        ctx_mask = utils.length2mask(lengths)

        # Decoder
        words = []
        log_probs = []
        hidden_states = []
        entropies = []
        h_t = torch.zeros(1, batch_size, self.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, self.rnn_dim).cuda()
        ended = np.zeros(len(obs), np.bool)
        start_token = '<NAV>' if for_nav else '<ORA>' # First word
        word = np.ones(len(obs), np.int64) * self.env.tok.word_to_index[start_token]    
        word = torch.from_numpy(word).view(-1, 1).cuda()
        # target = torch.LongTensor(len(obs)).cuda()
        stacked_logits = []
        for i in range(self.maxDecode):
            # Decode Step
            logits, h_t, c_t = self.decoder(word.long(), ctx, ctx_mask, h_t, c_t)      # Decode, logits: (b, 1, vocab_size)

            # Select the word
            logits = logits.squeeze()                                           # logits: (b, vocab_size)
            if insts is not None:
                # loss += self.softmax_loss(logits, insts[:,i].squeeze())
                stacked_logits.append(logits.clone().unsqueeze(2))
            logits[:, self.env.tok.word_to_index['<UNK>']] = -float('inf')         # No <UNK> in infer
            logits[:, self.env.tok.word_to_index['<PAD>']] = -float('inf')         # No <UNK> in infer
            
            if sampling:
                probs = F.softmax(logits, -1)
                m = torch.distributions.Categorical(probs)
                word = m.sample
                log_prob = m.log_prob(word)
                if train:
                    log_probs.append(log_prob)
                    hidden_states.append(h_t.squeeze())
                    entropies.append(m.entropy())
                else:
                    log_probs.append(log_prob.detach())
                    hidden_states.append(h_t.squeeze().detach())
                    entropies.append(m.entropy().detach())
            else:
                word = utils.next_decoder_input(logits, feedback, self.temperature, words, batch_size).int()
                # values, word = logits.max(1)

            # Append the word
            cpu_word = word.cpu().numpy()
            cpu_word[ended] = self.env.tok.word_to_index['<PAD>']
            words.append(cpu_word)

            # Prepare the shape for next step
            word = word.view(-1, 1)

            # End?
            ended = np.logical_or(ended, cpu_word == self.env.tok.word_to_index['<EOS>'])
            if ended.all():
                break

        if train and sampling:
            return np.stack(words, 1), torch.stack(log_probs, 1), torch.stack(hidden_states, 1), torch.stack(entropies, 1)
        elif train:
            p = torch.stack(stacked_logits,2).squeeze(3)[:,:,:-1]
            # print p.size()
            # print insts.size()
            seq_tensor = np.zeros((len(obs), self.env.tok.vocab_size(), insts.size()[1]), dtype=np.float32)
            seq_tensor[:, self.env.tok.word_to_index['<PAD>'], :] = 1.0
            seq_tensor = torch.from_numpy(seq_tensor).cuda()
            seq_tensor[:,:,:p.size()[2]] = p[:,:,:insts.size()[1]] 
            # print seq_tensor.size()
            # print np.stack(words, 1)
            return np.stack(words, 1), self.softmax_loss(seq_tensor, insts)#loss * 1.0/self.maxDecode
        else:
            # print "Predicted:"
            # print words
            return np.stack(words, 1), None      # [(b), (b), (b), ...] --> [b, l]

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        if not os.path.isdir(the_dir):
            os.makedirs(the_dir)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)
        def recover_state(name, model, optimizer, loadOptim=True):
            # print(name)
            # print(list(model.state_dict().keys()))
            # for key in list(model.state_dict().keys()):
            #     print(key, model.state_dict()[key].size())
            state = model.state_dict()
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        epoch = states['encoder']['epoch'] - 1
        print("Load the speaker's state dict from %s at epoch %d" % (path, epoch) )   
        return epoch

