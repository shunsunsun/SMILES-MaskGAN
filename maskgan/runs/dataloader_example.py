from collections import namedtuple, defaultdict

import random
import torch

from utils.args import maskgan_parser
from dataloaders import make_data_loader


def main():

    parser = maskgan_parser()
    args = parser.parse_args()
    kwargs = {'pin_memory': False, 'num_workers': 4}
    train_loader, val_loader, test_loader, vocab = make_data_loader(args, **kwargs)

    for i, samples in enumerate(train_loader):
        srcs, tgts, lengths, mask = samples
        tmp = srcs
        # print(f'src: {srcs}'
        #       f'tgts: {tgts}'
        #       f'lengths: {lengths}'
        #       f'mask: {mask}')


if __name__ == '__main__':
    main()
