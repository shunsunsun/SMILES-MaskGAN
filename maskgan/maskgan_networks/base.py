import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import uuid

from typing import Dict, List, Optional, Tuple

DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5


"""Refer to https://github.com/pytorch/fairseq"""
def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


def convert_padding_direction(
    src_tokens, padding_idx,
        right_to_left: bool = False,
        left_to_right: bool = False):

    assert right_to_left ^ left_to_right

    pad_mask = src_tokens.eq(padding_idx)
    if not pad_mask.any():
        # no padding, return early
        return src_tokens

    if left_to_right and not pad_mask[:, 0].any():
        # already right padded
        return src_tokens

    if right_to_left and not pad_mask[:, -1].any():
        # already left padded
        return src_tokens

    max_len = src_tokens.size(1)
    buffered = torch.empty(0).long()

    if max_len > 0:
        torch.arange(max_len, out=buffered)

    range = buffered.type_as(src_tokens).expand_as(src_tokens)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)

    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)

    else:
        index = torch.remainder(range + num_pads, max_len)

    return src_tokens.gather(1, index)


class LSTMEncoder(nn.Module):
    def __init__(self, args, dictionary,
                 bidirectional=False,
                 left_pad=False,
                 padding_idx=None,
                 pretrained_emb=None,
                 max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS):

        super(LSTMEncoder, self).__init__()
        self.args = args
        self.dictionary = dictionary
        self.bidirectional = bidirectional
        self.left_pad = left_pad
        self.padding_idx = padding_idx \
            if padding_idx is not None else dictionary['<pad>']

        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.enc_emb_dim = args.enc_emb_dim
        self.enc_hid_dim = args.enc_hid_dim
        self.max_source_positions = max_source_positions

        if pretrained_emb is None:
            self.embed_tokens = Embedding(len(dictionary), self.enc_emb_dim, self.padding_idx)

        else:
            self.embed_tokens = pretrained_emb

        self.rnn = LSTM(
            input_size=self.enc_emb_dim,
            hidden_size=self.enc_hid_dim,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True
        )

        if self.bidirectional:
            self.rnn_output_dim = self.enc_hid_dim * 2

        else:
            self.rnn_output_dim = self.enc_hid_dim

    def forward(self, src_tokens: Tensor,
                src_lengths: Tensor,
                enforce_sorted: bool = True):
        """
        params src_tokens: (batch_size, seq_len)
        params src_lengths: length per batch_size
        """

        # pack_padded_sequence requires right-padding
        # Convert left-padding to right-padding
        if self.left_pad:
            src_tokens = convert_padding_direction(
                src_tokens,
                torch.zeros_like(src_tokens).fill_(self.padding_idx),
                left_to_right=True)

        # (Batch, length)
        batch_size, seq_len = src_tokens.size()

        # embedding tokens
        embedded = self.embed_tokens(src_tokens)

        # dropout
        embedded = F.dropout(embedded, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        embedded = embedded.transpose(0, 1)

        # pack embedded source tokens into a packed sequence / batch_first = False
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), enforce_sorted=enforce_sorted)  # FIXME: lengths cuda:0 -> cpu

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, batch_size, self.enc_hid_dim

        else:
            state_size = self.num_layers, batch_size, self.enc_hid_dim

        h0 = embedded.new_zeros(*state_size)
        c0 = embedded.new_zeros(*state_size)

        packed_outputs, (final_hiddens, final_cells) = self.rnn(packed, (h0, c0))

        # unpack outputs and apply dropout / batch_first = False
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, padding_value=self.padding_idx * 1.0)
        encoder_outputs = F.dropout(encoder_outputs, p=self.dropout, training=self.training)

        # assert list(encoder_outputs.size()) == [seq_len, batch_size, self.rnn_output_dim]  # FIXME

        if self.bidirectional:
            final_hiddens = self.combine_bidir(final_hiddens, batch_size)
            final_cells = self.combine_bidir(final_cells, batch_size)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return tuple((
            encoder_outputs,  # seq_len x batch_size x hidden
            final_hiddens,  # num_layers x batch_size x num_directions * hidden
            final_cells,  # num_layers x batch_size x num_directions * hidden
            encoder_padding_mask  # seq_len x batch_size
        ))

    def combine_bidir(self, outputs, batch_size: int):
        out = outputs.view(self.num_layers, 2, batch_size, -1).transpose(1, 2).contiguous()

        return out.view(self.num_layers, batch_size, -1)


class Attention(nn.Module):
    def __init__(self, args, source_embed_dim, bias=False):
        super(Attention, self).__init__()

        self.args = args
        self.input_embed_dim = args.dec_hid_dim
        self.source_embed_dim = source_embed_dim
        self.output_embed_dim = args.dec_hid_dim

        self.input_proj = Linear(
            self.input_embed_dim, self.source_embed_dim, bias=bias)
        self.output_proj = Linear(
            self.input_embed_dim + self.source_embed_dim,
            self.output_embed_dim, bias=bias)

    def forward(self, input, source_hidden, encoder_padding_mask):
        """
        input: batch_size x self.input_embed_dim
        source_hidden: source length x batch_size x source_embed_dim
        """

        # x: batch_size x source_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hidden * x.unsqueeze(0)).sum(dim=2)

        # do not attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask, float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # source length x batch_size

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hidden).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))

        return x, attn_scores


class LSTMDecoder(nn.Module):
    """LSTM decoder"""
    def __init__(self, args, dictionary, encoder_output,
                 attention_module, pretrained_emb=None,
                 max_target_positions=DEFAULT_MAX_TARGET_POSITIONS):
        super(LSTMDecoder, self).__init__()

        self._incremental_state_id = str(uuid.uuid4())

        self.args = args
        self.dictionary = dictionary
        self.encoder_output = encoder_output
        self.max_target_positions = max_target_positions

        self.need_attn = True
        self.adaptive_softmax = None

        self.dropout = args.dropout
        self.dec_emb_dim = args.dec_emb_dim
        self.dec_hid_dim = args.dec_hid_dim
        self.out_emb_dim = args.dec_out_emb_dim
        self.num_layers = args.num_layers
        self.share_input_output_embed = args.share_dec_input_output_emb

        padding_idx = dictionary['<pad>']

        if pretrained_emb is None:
            self.embed_tokens = Embedding(len(dictionary), self.dec_emb_dim, padding_idx)

        else:
            self.embed_tokens = pretrained_emb

        if self.encoder_output != self.dec_hid_dim and self.encoder_output != 0:
            self.encoder_hidden_proj = Linear(self.encoder_output, self.dec_hid_dim)
            self.encoder_cell_proj = Linear(self.encoder_output, self.dec_hid_dim)

        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        # Disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if self.encoder_output == 0 else self.dec_hid_dim

        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=input_feed_size + self.dec_emb_dim if layer == 0 else self.dec_hid_dim,
                hidden_size=self.dec_hid_dim)
            for layer in range(self.num_layers)])

        if isinstance(attention_module, nn.Module):
            self.attention = Attention(args, self.encoder_output, bias=False)

        else:
            self.attention = None

        if self.dec_hid_dim != self.out_emb_dim:
            self.additional_fc = Linear(self.dec_hid_dim, self.out_emb_dim)

        if not self.share_input_output_embed:
            self.fc_out = Linear(self.out_emb_dim, len(dictionary), dropout=self.dropout)

    def get_cached_state(self,
                         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]):
        cached_state = self.get_incremental_state(incremental_state, 'cached_state')
        assert cached_state is not None

        prev_hiddens = cached_state['prev_hiddens']
        assert prev_hiddens is not None

        prev_cells = cached_state['prev_cells']
        assert prev_cells is not None

        prev_hiddens = [prev_hiddens[i] for i in range(self.num_layers)]
        prev_cells = [prev_cells[j] for j in range(self.num_layers)]

        input_feed = cached_state['input_feed']

        return prev_hiddens, prev_cells, input_feed

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state

    def forward(self, prev_output_tokens,
                encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
                incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                src_lengths: Optional[Tensor] = None):

        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )

        return self.output_layer(x), attn_scores

    def extract_features(self, prev_output_tokens,
                         encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
                         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None):

        """
        Similar to *forward* but only return features
        """

        # Get outputs from encoder
        if encoder_out is not None:
            encoder_outs = encoder_out[0]
            encoder_hiddens = encoder_out[1]
            encoder_cells = encoder_out[2]
            encoder_padding_mask = encoder_out[3]

        else:
            encoder_outs = torch.empty(0)
            encoder_hiddens = torch.empty(0)
            encoder_cells = torch.empty(0)
            encoder_padding_mask = torch.empty(0)

        src_len = encoder_outs.size(0)

        if incremental_state is not None and len(incremental_state) > 0:
            prev_output_tokens = prev_output_tokens[:, -1:]

        batch_size, seq_len = prev_output_tokens.size()

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Initialize previous states (or get from cache during incremental generation)
        if incremental_state is not None and len(incremental_state) > 0:
            prev_hiddens, prev_cells, input_feed = self.get_cached_state(incremental_state)

        elif encoder_out is not None:
            # setup recurrent cells
            prev_hiddens = [encoder_hiddens[i] for i in range(self.num_layers)]
            prev_cells = [encoder_cells[j] for j in range(self.num_layers)]

            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(y) for y in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(y) for y in prev_cells]

            input_feed = x.new_zeros(batch_size, self.dec_hid_dim)

        else:
            # setup zero cells, since there is no encoder
            zero_state = x.new_zeros(batch_size, self.dec_hid_dim)
            prev_hiddens = [zero_state for i in range(self.num_layers)]
            prev_cells = [zero_state for i in range(self.num_layers)]
            input_feed = None

        assert src_len > 0 or self.attention is None, \
            "attention is not supported if there are no encoder outputs"

        attn_scores = x.new_zeros(src_len, seq_len, batch_size) if self.attention is not None else None

        outs = []
        for j in range(seq_len):
            # Input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed), dim=1)

            else:
                input = x[j]

            for i, rnn in enumerate(self.layers):
                # Recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                assert attn_scores is not None
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)

            else:
                out = hidden

            out = F.dropout(out, p=self.dropout, training=self.training)

            # input feeding
            if input_feed is not None:
                input_feed = out

            # save final output
            outs.append(out)

        # Stack all the necessary tensors together and store
        prev_hiddens_tensor = torch.stack(prev_hiddens)
        prev_cells_tensor = torch.stack(prev_cells)

        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {"prev_hiddens": prev_hiddens_tensor, "prev_cells": prev_cells_tensor, "input_feed": input_feed})

        self.set_incremental_state(
            incremental_state, 'cached_state', cache_state)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seq_len, batch_size, self.dec_hid_dim)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # src_len x tgt_len x batch_size -> batch_size x tgt_len x src_len
        if not self.training and self.need_attn and self.attention is not None:
            assert attn_scores is not None
            attn_scores = attn_scores.transpose(0, 2)

        else:
            attn_scores = None

        return x, attn_scores

    def output_layer(self, x):
        """
        Project features to the vocabulary size
        """

        if self.adaptive_softmax is None:

            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)

            else:
                x = self.fc_out(x)

        return x


class LSTMModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(LSTMModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        assert isinstance(self.encoder, nn.Module)
        assert isinstance(self.decoder, nn.Module)

    @classmethod
    def build_model(cls, args, task):

        if task.source_dictionary != task.target_dictionary:

            raise ValueError

        if args.enc_emb_dim != args.dec_emb_dim:
            raise ValueError

        if args.share_dec_input_output_emb and (
            args.dec_emb_dim != args.dec_out_emb_dim):

            raise ValueError

        encoder = LSTMEncoder(
            args=args,
            dictionary=task.source_dictionary,
            bidirectional=args.bidirectional)

        decoder = LSTMDecoder(
            args=args,
            dictionary=task.target_dictionary,
            encoder_output=encoder.rnn_output_dim,
            attention_module=Attention)

        return cls(encoder, decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens,
                incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None):

        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out,
            incremental_state=incremental_state
        )

        return decoder_out
