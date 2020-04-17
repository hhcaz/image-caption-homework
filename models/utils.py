import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ComposedEmbedding(nn.Module):
    def __init__(self, pretrained_size, trainable_size, embedding_dim):
        super(ComposedEmbedding, self).__init__()
        self.pretrained_size = pretrained_size
        self.trainable_size = trainable_size
        self.pretrained = nn.Embedding(pretrained_size, embedding_dim)
        self.trainable = nn.Embedding(trainable_size, embedding_dim)
        
        self.fine_tune_pretrained(False)
        self.fine_tune_trainable(True)

        print('pretrained:', self.pretrained.weight.shape)
        print('trainable: ', self.trainable.weight.shape)
    
    @property
    def vocab_size(self):
        return self.pretrained_size + self.trainable_size
    
    def load_pretained(self, tensor):
        self.pretrained.weight = nn.Parameter(tensor, requires_grad=False)
        self.pretrained.weight.requires_grad = False
    
    def fine_tune_pretrained(self, yes=False):
        for p in self.pretrained.parameters():
            p.requires_grad = yes
    
    def fine_tune_trainable(self, yes=True):
        for p in self.trainable.parameters():
            p.requires_grad = yes
    
    def fine_tune_all(self, yes):
        self.fine_tune_pretrained(yes)
        self.fine_tune_trainable(yes)
    
    def forward(self, idx):
        mask = idx >= self.pretrained_size
        pretrained_idx = idx.clone().detach()
        pretrained_idx[mask] = 0
        trainable_idx = idx - self.pretrained_size
        
        embeddings = self.pretrained(pretrained_idx)
        # print('requires_grad (before mask)?', embeddings.requires_grad)
        embeddings[mask] = self.trainable(trainable_idx[mask])
        # print('requires_grad (after mask)?', embeddings.requires_grad)

        return embeddings


class BeamSearcher(object):
    def __init__(self, bos_index, eos_index, beam_size=5, max_steps=50, device=torch.device("cuda:0")):
        """
        Arguments:
        - bos_index: word index of <bos>
        - eos_index: word index of <eos>
        - beam_size: beam size
        - max_steps: max caption lengths, exclude <bos> and <eos>
        """
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.beam_size = beam_size
        self.max_steps = max_steps
        self.device    = device

        self.extra_sequences = {}
        self.extra_complete_sequences = {}

        self.reset()

    def reset(self, remove_tracers=True):
        """
        Arguments:
        - remove_tracers: bool, if True, all the previous registered history tracers will be removed, else only empty them.
        """
        # # All branches start with <bos>
        # start_words_idx = torch.LongTensor(self.beam_size).fill_(self.bos_index).to(self.device)

        # # Sequences keep the history for top_k_prev_words_idx
        # # shape=(beam_size, time_steps), now time_steps=1
        # self.sequences = start_words_idx.unsqueeze(dim=-1) # shape=(beam_size, 1)
        self.sequences = None

        # Tensor to store top k sequences' scores; now they're just 0
        # we set the score after the first to be -inf so that the first beam would be selected anyway
        self.top_k_scores = torch.zeros(self.beam_size).to(self.device)
        self.top_k_scores[1:].fill_(float('-inf'))

        self.time_steps = 1

        # Beam size will be updated for each time_step, since some branches will hit <eos>;
        # now is original beam size 
        self.beam_size_t = self.beam_size 

        self.complete_sequences = []
        self.complete_sequences_scores = []
        
        self.search_started = False
        self.search_terminated = True

        # Clear the user registered sequences
        if remove_tracers:
            self.extra_sequences.clear()
            self.extra_complete_sequences.clear()
        else:
            for name in self.registered_tracer_names:
                self.extra_sequences[name] = None
                self.extra_complete_sequences[name] = []

        return self
    
    def start(self):
        """Prepare for the beam search and returns tensor with shape (beam_size,) filled with index of <bos>."""
        assert not self.is_started, "Beam search has already started, please call .`reset()` first"
        self.search_started = True
        self.search_terminated = False

        # All branches start with <bos>
        start_words_idx = torch.LongTensor(self.beam_size).fill_(self.bos_index).to(self.device)

        return start_words_idx

    def register_history_tracer(self, name):
        assert name not in self.extra_sequences.keys(), "Tracer {} already registered.".format(name)
        assert not self.is_started, "Beam search already started, register before calling `.start()`"
        self.extra_sequences[name] = None
        self.extra_complete_sequences[name] = []

    def remove_history_tracer(self, name):
        assert name in self.extra_sequences.keys(), "Tracer {} hasn't been registered.".format(name)
        self.extra_sequences.pop(name)
        self.extra_complete_sequences.pop(name)

    def empty_history_tracer(self, name):
        assert name in self.extra_sequences.keys(), "Tracer {} hasn't been registered.".format(name)
        assert self.is_terminated, "Beam search not terminated, cannot empty history tracer yet."
        self.extra_sequences[name] = None
        self.extra_complete_sequences[name] = []

    @property
    def registered_tracer_names(self):
        return list(self.extra_sequences.keys())

    def step(self, instant_scores, **tracer_contents):
        """
        Arguments:
        - instant_scores: shape=(beam_size_t, vocab_size), raw predictions (not softmaxed) from decoder

        Returns:
        - prev_beam_idx: shape=(beam_size_t,), value range in [0, beam_size_t), telling which previous branches are selected
        - next_word_idx: shape=(beam_size_t,), value range in [0, vocab_size), telling which words are selected
        - incomplete_beam_idx: bool tensor, shape=(beam_size_t_new,), beam_size_t_new is smaller than or equal to beam_size_t
            Note: incomplete_beam_idx tells which branches havn't met <eos>, when next time user calls this function,
            the shape of `instant_scores` should be (beam_size_t_new, vocab_size). This means user should filter
            some variables using `incomplete_beam_idx` when calculating `instant_scores`

        Note: when in time_step t, assume the shape of hidden state h<t> as (beam_size_t, decoder_dim), after calling
        `.step()`, the common process for h<t> is h<t+1> := h<t>[prev_beam_idx][incomplete_beam_idx]
        (or h<t> := h<t>[prev_beam_idx[incomplete_beam_idx]], they are just the same, and the latter is more efficient),
        thus the shape of selected hidden state h<t+1> will be (beam_size_t_new,); the same process should also be applied
        to hidden state c<t> and other variables which contribute to `instant_scores`.
        """
        assert not self.is_terminated, "Search has been terminated, please call `.reset()` first."
        assert self.is_started, "Search hasn't been started, please call `.start()` first."
        assert set(self.registered_tracer_names) <= set(tracer_contents.keys()), \
            "Not enough contents provided for all the registered trancers. " \
            + "Required contents for {} but received {}.".format(set(self.registered_tracer_names), set(tracer_contents.keys()))

        beam_size_t, vocab_size = instant_scores.size()
        assert beam_size_t == self.beam_size_t, \
            "Beam size has been updated, expect {} but receive {}".format(self.beam_size_t, beam_size_t)

        # Log + softmax the raw scores
        instant_scores = F.log_softmax(instant_scores, dim=-1) # (beam_size_t, vocab_size)
        
        # Calculate accumulated scores, i.e. log P(x, y<1>, y<2>, ..., y<t>)
        accumulated_scores = self.top_k_scores.unsqueeze(-1) + instant_scores # shape=(beam_size_t, vocab_size)

        # Select top k scores, from all the branches, so we need to unroll accumulated_scores to find top scores
        # accumulated_scores = accumulated_scores.view(beam_size_t * vocab_size)
        self.top_k_scores, top_k_words_idx = accumulated_scores.view(-1).topk(k=beam_size_t, dim=-1) # shape=(beam_size_t,)

        # Convert unrolled indices to actual indices
        prev_beam_idx = top_k_words_idx / vocab_size # shape=(beam_size_t,), value range in [0, beam_size_t)
        next_word_idx = top_k_words_idx % vocab_size # shape=(beam_size_t,), value range in [0, vocab_size)

        # Update sequences
        if self.sequences is None:
            self.sequences = next_word_idx.unsqueeze(1)
        else:
            # selected_prev_sequences = sequences[prev_beam_idx]
            self.sequences = torch.cat((self.sequences[prev_beam_idx], next_word_idx.unsqueeze(1)), dim=1)
            # shape from (beam_size_t, time_steps) -> (beam_size_t, time_steps+1)

        # Update registered tracers' content
        for name in self.registered_tracer_names:
            if self.extra_sequences[name] is None:
                self.extra_sequences[name] = tracer_contents[name].unsqueeze(1)
            else:
                self.extra_sequences[name] = torch.cat(
                    (self.extra_sequences[name][prev_beam_idx], tracer_contents[name].unsqueeze(1)),
                    dim=1
                )

        # Set true if some braches hit the <eos>
        complete_beam_mask = next_word_idx == self.eos_index # shape=(beam_size_t,)
        incomplete_beam_mask = ~complete_beam_mask # shape=(beam_size_t,)
        incomplete_beam_idx = torch.nonzero(incomplete_beam_mask, as_tuple=True)[0] # shape=(beam_size_t_new,)

        # If some braches hit the <eos>
        if torch.sum(complete_beam_mask) > 0:
            self.complete_sequences.extend(self.sequences[complete_beam_mask].tolist()) # (beam_size - beam_size_t_new, seq_len)
            self.complete_sequences_scores.extend(self.top_k_scores[complete_beam_mask].tolist()) # (beam_size - beam_size_t_new,)

            # Update registered tracer's content
            for name in self.registered_tracer_names:
                self.extra_complete_sequences[name].extend(self.extra_sequences[name][complete_beam_mask])
                self.extra_sequences[name] = self.extra_sequences[name][incomplete_beam_idx]
        
        self.sequences = self.sequences[incomplete_beam_idx] # (beam_size_t_new, time_steps+1)
        self.top_k_scores = self.top_k_scores[incomplete_beam_idx] # (beam_size_t_new,)

        self.beam_size_t = len(incomplete_beam_idx) # update beam_size_t, beam_size_t := beam_size_t_new
        if self.beam_size_t == 0 or self.time_steps > self.max_steps:
            self.search_terminated = True
        
        self.time_steps += 1
        
        return prev_beam_idx, next_word_idx, incomplete_beam_idx

    def force_terminate(self):
        self.search_terminated = True
        return self

    @property
    def is_started(self):
        return self.search_started

    @property
    def is_terminated(self):
        return self.search_terminated

    def fetch_results(self, return_best=True, desending_order=True):
        """
        Arguments:
        - return_best: if True the returns the results of branch which has the highest score else return results of all the braches
        - descending_order: if True then sort the results accroding to branches' scores in descending order else ascending order.
            Only work when `return_best` is False
        
        Returns:
        - caption(s): word indexes of caption(s), (include <bos> and <eos> but no <pad>)
        - score(s): score(s) of returned caption(s)
        - registered_tracer_content(s): sequence content(s) of registered tracers
        """
        assert self.is_terminated, "Beam search hasn't terminated, call `.force_terminate()` if you still want to get captions."

        if return_best:
            idx = np.argmax(self.complete_sequences_scores)
            caption = self.complete_sequences[idx]
            score = self.complete_sequences_scores[idx]

            registered_tracer_content = {}
            for name in self.registered_tracer_names:
                registered_tracer_content[name] = self.extra_complete_sequences[name][idx]
            
            return caption, score, registered_tracer_content

        else:
            sort_idx = np.argsort(self.complete_sequences_scores)
            
            if desending_order:
                sort_idx = sort_idx[::-1]

            captions = [self.complete_sequences[i] for i in sort_idx]
            scores = [self.complete_sequences_scores[i] for i in sort_idx]

            registered_tracer_contents = {}
            for name in self.registered_tracer_names:
                registered_tracer_contents[name] = [self.extra_complete_sequences[name][i] for i in sort_idx]
            
            return captions, scores, registered_tracer_contents
