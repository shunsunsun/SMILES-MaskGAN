import os
import pickle

from copy import deepcopy

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils.mypath import data_path
from maskgan.utils.vocab_builder import VocabBuilder


class MaskSmilesDataset(Dataset):
    def __init__(self, args, mask_builder, base_dir=data_path, vocab=None, split='train'):

        self.args = args
        self.mask_builder = mask_builder
        self.base_dir = base_dir
        self.vocab = vocab
        self.split = split
        self._construct_vocabulary()

        self.encode_dict = {"Br": 'Y', "Cl": 'X', "Si": 'A', 'Se': 'Z', '@@': 'R', 'se': 'E'}
        self.decode_dict = {v: k for k, v in self.encode_dict.items()}

        if self.split == 'train':
            with open(os.path.join(base_dir, f'chembl26_canon_train_{args.probability}.pkl'), 'rb') as f:
                train_data = pickle.load(f)

                self.train_srcs = train_data['train_srcs']
                self.train_tgts = train_data['train_tgts']
                self.train_lengths = train_data['train_lengths']
                self.train_mask = train_data['train_mask']

        elif self.split == 'val':
            with open(os.path.join(base_dir, f'chembl26_canon_valid_{args.probability}.pkl'), 'rb') as f:
                val_data = pickle.load(f)

                self.val_srcs = val_data['val_srcs']
                self.val_tgts = val_data['val_tgts']
                self.val_lengths = val_data['val_lengths']
                self.val_mask = val_data['val_mask']

        elif self.split == 'test':
            with open(os.path.join(base_dir, f'chembl26_canon_test_{args.probability}.pkl'), 'rb') as f:
                test_data = pickle.load(f)

                self.test_srcs = test_data['test_srcs']
                self.test_tgts = test_data['test_tgts']
                self.test_lengths = test_data['test_lengths']
                self.test_mask = test_data['test_mask']

        else:
            raise NotImplementedError

    def encode(self, smiles: str) -> str:
        """
        Replace multi-char tokens with single tokens in SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            sanitized SMILE string with only single-char tokens
        """

        temp_smiles = smiles
        for symbol, token in self.encode_dict.items():
            temp_smiles = temp_smiles.replace(symbol, token)
        return temp_smiles

    def _construct_vocabulary(self):

        if self.vocab is None:
            builder = VocabBuilder(self.mask_builder)
            self.vocab = builder.vocab()

    def __len__(self):

        if self.split == 'train':

            return len(self.train_srcs)

        elif self.split == 'val':

            return len(self.val_srcs)

        elif self.split == 'test':
            
            return len(self.test_srcs)

    def __getitem__(self, idx):

        if self.split == 'train':
            srcs = self.train_srcs[idx]
            tgts = self.train_tgts[idx]
            lengths = self.train_lengths[idx]
            mask = self.train_mask[idx]

            return srcs, tgts, lengths, mask

        elif self.split == 'val':
            srcs = self.val_srcs[idx]
            tgts = self.val_tgts[idx]
            lengths = self.val_lengths[idx]
            mask = self.val_mask[idx]

            return srcs, tgts, lengths, mask

        elif self.split == 'test':
            srcs = self.test_srcs[idx]
            tgts = self.test_tgts[idx]
            lengths = self.test_lengths[idx]
            mask = self.test_mask[idx]

            return srcs, tgts, lengths, mask

        else:
            raise NotImplementedError

    def get_collate_fn(self):

        return MaskSmilesDataset.collate

    @staticmethod
    def collate(samples):
        srcs, tgts, lengths, masks = list(zip(*samples))

        srcs = torch.LongTensor(srcs)  # LongTensor == torch.int64
        tgts = torch.LongTensor(tgts)

        lengths = torch.LongTensor(lengths)
        lengths, sort_order = lengths.sort(descending=True)

        def _rearrange(tensor):

            return tensor.index_select(0, sort_order)

        srcs = _rearrange(pad_sequence(srcs, batch_first=True, padding_value=1))
        tgts = _rearrange(pad_sequence(tgts, batch_first=True, padding_value=1))
        masks = _rearrange(torch.stack(masks, dim=0))

        return srcs, tgts, lengths, masks
