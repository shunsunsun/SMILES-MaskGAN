import torch
import pdb
import math


def greed_sample(logits):
    batch_size, seq_len, _ = logits.size()
    max_values, max_indices = logits.max(dim=2)

    return max_indices


def ppl(sequences, log_probs):
    batch_size, seq_len = sequences.size()
    seq_log_probs = torch.zeros_like(sequences).float()

    for b in range(batch_size):
        for t in range(seq_len):
            idx = sequences[b, t].item()
            seq_log_probs[b, t] = log_probs[b, t, idx].item()

    return seq_log_probs.sum()


def perplexity(truths, sampled, log_probs):
    batch_size, seq_len, vocab_size = log_probs.size()

    _ppl = {
        'ground-truth': ppl(truths, log_probs).mean(),
        'sampled': ppl(sampled, log_probs).mean()}

    return _ppl
