import os

import torch
from utils.mypath import data_path


if not os.path.exists(os.path.join(data_path, 'vocab')):
    os.makedirs(os.path.join(data_path, 'vocab'))


class VocabBuilder:
    def __init__(self, mask_builder):
        self.vocab_path = os.path.join(data_path, 'vocab', 'vocab.pt')

        self.mask_builder = mask_builder
        self._vocab = None

    def vocab(self):
        if self._vocab is None:
            self.build_vocab()

        return self._vocab

    def build_vocab(self):
        if os.path.exists(self.vocab_path):
            self._vocab = torch.load(self.vocab_path)

        else:
            self.rebuild_vocab()

    def rebuild_vocab(self):
        self.forbidden_symbols = {'Ag', 'Al', 'Am', 'Ar', 'At', 'Au', 'D', 'E', 'Fe', 'G', 'K', 'L', 'M', 'Ra', 'Re',
                                  'Rf', 'Rg', 'Rh', 'Ru', 'T', 'U', 'V', 'W', 'Xe',
                                  'Y', 'Zr', 'a', 'd', 'f', 'g', 'h', 'k', 'm', 'si', 't', 'te', 'u', 'v', 'y'}

        self._vocab = {'<unk>': 0, '<pad>': 1, '<eos>': 2, '#': 20, '%': 22, '(': 25, ')': 24, '+': 26, '-': 27,
                         '.': 30,
                         '0': 32, '1': 31, '2': 34, '3': 33, '4': 36, '5': 35, '6': 38, '7': 37, '8': 40,
                         '9': 39, '=': 41, 'A': 7, 'B': 11, 'C': 19, 'F': 4, 'H': 6, 'I': 5, 'N': 10,
                         'O': 9, 'P': 12, 'S': 13, 'X': 15, 'Y': 14, 'Z': 3, '[': 16, ']': 18,
                         'b': 21, 'c': 8, 'n': 17, 'o': 29, 'p': 23, 's': 28,
                         "@": 42, "R": 43, '/': 44, "\\": 45, 'E': 46, '__<m>__': 47
                         }

        self.idx_char = {v: k for k, v in self._vocab.items()}

        if self.vocab_path is not None:
            torch.save(self._vocab, self.vocab_path)
