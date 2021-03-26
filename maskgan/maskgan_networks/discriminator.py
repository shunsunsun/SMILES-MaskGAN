import torch
import torch.nn as nn
import torch.nn.functional as F

from maskgan.maskgan_networks.base import LSTMEncoder, Attention, LSTMDecoder


class MaskGANDiscriminator(nn.Module):
    def __init__(self, args, task):
        super(MaskGANDiscriminator, self).__init__()

        self.args = args

        if task.source_dictionary != task.target_dictionary:

            raise ValueError

        if args.enc_emb_dim != args.dec_emb_dim:
            raise ValueError

        if args.share_dec_input_output_emb and (
            args.dec_emb_dim != args.dec_out_emb_dim):

            raise ValueError

        pretrained_enc_emb = nn.Embedding(
            len(task.source_dictionary), args.enc_emb_dim, task.source_dictionary['<pad>'])

        pretrained_dec_emb = None

        self.encoder = LSTMEncoder(
            args=args,
            dictionary=task.source_dictionary,
            bidirectional=args.bidirectional,
            pretrained_emb=pretrained_enc_emb)

        self.decoder = LSTMDecoder(
            args=args,
            dictionary=task.target_dictionary,
            encoder_output=self.encoder.rnn_output_dim,
            attention_module=Attention,
            pretrained_emb=pretrained_dec_emb)

        out_embed_dim = self.decoder.additional_fc.out_feature \
            if hasattr(self, 'additional_fc') else self.decoder.dec_hid_dim

        self.decoder.fc_out = nn.Linear(out_embed_dim, 1)

        assert isinstance(self.encoder, nn.Module)
        assert isinstance(self.decoder, nn.Module)

    def forward(self, srcs, lengths, samples):
        self.encoder.rnn.flatten_parameters()

        encoder_output = self.encoder(srcs, lengths)
        x, attn_scores = self.decoder(samples, encoder_out=encoder_output)

        return x, attn_scores
