import os
import tqdm
import pickle
import pandas as pd

from copy import deepcopy

import torch
from collections import defaultdict

from utils.mypath import data_path, save_path
from maskgan.utils.vocab_builder import VocabBuilder
from maskgan.utils.mask import StochasticMask


torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

encode_dict = {"Br": 'Y', "Cl": 'X', "Si": 'A', 'Se': 'Z', '@@': 'R', 'se': 'E'}
decode_dict = {v: k for k, v in encode_dict.items()}

probability = 0.3
rmask = StochasticMask(probability)

vocab = None
if vocab is None:
    builder = VocabBuilder(rmask)
    vocab = builder.vocab()


def refined_data(path, prob, split: str = 'train'):
    if split == 'train':
        train_dir = os.path.join(path, 'chembl26_canon_train.csv')
        train_ = pd.read_csv(train_dir)
        train_list = [line.strip() for line in train_['canonical_smiles']]
        train_list = [encode(line) for line in train_list]

        train_data = defaultdict(
            list, {
                k: [] for k in ('train_srcs', 'train_tgts', 'train_lengths', 'train_mask')
            }
        )

        for i, tokens in enumerate(train_list):
            seq_len = len(tokens)
            mask_idxs = rmask(seq_len)
            mask_id = vocab['__<m>__']

            src, tgt, length, mask = get_pair(tokens, mask_idxs, mask_id)

            train_data['train_srcs'].append(src)
            train_data['train_tgts'].append(tgt)
            train_data['train_lengths'].append(length)
            train_data['train_mask'].append(mask)

        with open(os.path.join(data_path, f'chembl26_canon_train_{prob}.pkl'), 'wb') as f:
            pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)

    elif split == 'val':
        val_dir = os.path.join(path, 'chembl26_canon_valid.csv')
        val_ = pd.read_csv(val_dir)
        val_list = [line.strip() for line in val_['canonical_smiles']]
        val_list = [encode(line) for line in val_list]

        val_data = defaultdict(
            list, {
                k: [] for k in ('val_srcs', 'val_tgts', 'val_lengths', 'val_mask')
            }
        )

        for i, tokens in enumerate(val_list):
            seq_len = len(tokens)
            mask_idxs = rmask(seq_len)
            mask_id = vocab['__<m>__']

            src, tgt, length, mask = get_pair(tokens, mask_idxs, mask_id)

            val_data['val_srcs'].append(src)
            val_data['val_tgts'].append(tgt)
            val_data['val_lengths'].append(length)
            val_data['val_mask'].append(mask)

        with open(os.path.join(data_path, f'chembl26_canon_valid_{prob}.pkl'), 'wb') as f:
            pickle.dump(val_data, f, pickle.HIGHEST_PROTOCOL)

    elif split == 'test':
        test_dir = os.path.join(path, 'chembl26_canon_all.csv')
        test_ = pd.read_csv(test_dir)
        test_list = [line.strip() for line in test_['canonical_smiles']]
        test_list = [encode(line) for line in test_list]

        test_data = defaultdict(
            list, {
                k: [] for k in ('test_srcs', 'test_tgts', 'test_lengths', 'test_mask')
            }
        )

        for i, tokens in enumerate(test_list):
            seq_len = len(tokens)
            mask_idxs = rmask(seq_len)
            mask_id = vocab['__<m>__']

            src, tgt, length, mask = get_pair(tokens, mask_idxs, mask_id)

            test_data['test_srcs'].append(src)
            test_data['test_tgts'].append(tgt)
            test_data['test_lengths'].append(length)
            test_data['test_mask'].append(mask)

        with open(os.path.join(data_path, f'chembl26_canon_test_{prob}.pkl'), 'wb') as f:
            pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)


def encode(smiles: str) -> str:
    """
    Replace multi-char tokens with single tokens in SMILES string.

    Args:
        smiles: SMILES string

    Returns:
        sanitized SMILE string with only single-char tokens
    """

    temp_smiles = smiles
    for symbol, token in encode_dict.items():
        temp_smiles = temp_smiles.replace(symbol, token)
    return temp_smiles


def get_pair(tokens, mask_idxs, mask_id):
    idxs = [vocab[atom] for atom in tokens]

    def _pad(ls, pad_index, max_length=100):
        padded_ls = deepcopy(ls)

        while len(padded_ls) <= max_length:
            padded_ls.append(pad_index)

        return padded_ls

    srcs = deepcopy(idxs)
    srcs.append(vocab['<eos>'])  # append eos id in srcs last

    tgts = deepcopy(idxs)
    tgts.insert(0, vocab['<eos>'])  # insert eos id in tgts first
    # tgts.append(self.vocab['<eos>'])  # same as srcs, changed 20200715

    srcs = _pad(srcs, vocab['<pad>'], max_length=100)
    tgts = _pad(tgts, vocab['<pad>'], max_length=100)

    mask = torch.zeros(len(tgts))
    for mask_idx in mask_idxs:
        offset = 1
        mask[mask_idx + offset] = 1
        srcs[mask_idx] = mask_id

    return srcs, tgts, len(srcs), mask


with tqdm.tqdm(total=3) as tbar:
    for s in ['train', 'val', 'test']:
        refined_data(data_path, probability, split=s)
        tbar.update(1)

