from warnings import warn

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from maskgan.maskgan_networks.base import LSTMEncoder, LSTMDecoder, Attention


class MaskGANGenerator(nn.Module):
    def __init__(self, args, task):
        super(MaskGANGenerator, self).__init__()

        self.args = args

        if task.source_dictionary != task.target_dictionary:

            raise ValueError

        if args.enc_emb_dim != args.dec_emb_dim:
            raise ValueError

        if args.share_dec_input_output_emb and (
            args.dec_emb_dim != args.dec_out_emb_dim):

            raise ValueError

        self.encoder = LSTMEncoder(
            args=args,
            dictionary=task.source_dictionary,
            bidirectional=args.bidirectional)

        self.decoder = LSTMDecoder(
            args=args,
            dictionary=task.target_dictionary,
            encoder_output=self.encoder.rnn_output_dim,
            attention_module=Attention)

        assert isinstance(self.encoder, nn.Module)
        assert isinstance(self.decoder, nn.Module)

    def forward(self, srcs, lengths, tgts, mask):
        self.encoder.rnn.flatten_parameters()

        encoder_out = self.encoder(srcs, lengths)  # Tuple returned
        logits, attns = self.decoder(tgts, encoder_out=encoder_out,
                                     incremental_state=None, src_lengths=None)

        batch_size, seq_len, vocab_size = logits.size()

        samples, log_probs = [], []
        for t in range(seq_len):
            logit = logits[:, t, :]
            distribution = Categorical(logits=logit)
            sampled = distribution.sample()
            fsampled = torch.where(mask[:, t].byte(), sampled, srcs[:, t])
            log_prob = distribution.log_prob(fsampled)

            log_probs.append(log_prob)
            samples.append(fsampled)

        samples = torch.stack(samples, dim=1)
        log_probs = torch.stack(log_probs, dim=1)

        return samples, log_probs, attns

    def logits(self, srcs, lengths, tgts, mask):
        self.encoder.rnn.flatten_parameters()
        encoder_out = self.encoder(srcs, lengths)
        logits, attns = self.decoder(tgts, encoder_out=encoder_out)

        return logits
