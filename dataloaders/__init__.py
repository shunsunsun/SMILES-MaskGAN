import os

from utils.mypath import data_path

from dataloaders.dataset import masksmiles
from torch.utils.data import DataLoader

from maskgan.utils.mask import StochasticMask


def make_data_loader(args, mode='train', **kwargs):

    if args.dataset == 'masksmiles':

        if mode == 'train':
            # Define mask
            rmask = StochasticMask(probability=args.probability)

            train_set = masksmiles.MaskSmilesDataset(args, rmask, split='train')

            vocab = train_set.vocab
            val_set = masksmiles.MaskSmilesDataset(args, rmask, vocab=vocab, split='val')
            test_set = masksmiles.MaskSmilesDataset(args, rmask, vocab=vocab, split='test')

            train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                    collate_fn=masksmiles.MaskSmilesDataset.collate,
                                    shuffle=True, **kwargs)
            val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                                    collate_fn=masksmiles.MaskSmilesDataset.collate,
                                    shuffle=False, **kwargs)
            test_loader = None

            return train_loader, val_loader, test_loader, vocab
        
        else:
            rmask = StochasticMask(probability=args.probability)

            train_set = masksmiles.MaskSmilesDataset(args, rmask, split='train')
            vocab = train_set.vocab
            
            test_set = masksmiles.MaskSmilesDataset(args, rmask, vocab=vocab, split='test')
            
            train_loader = None
            val_loader = None
            test_loader = DataLoader(test_set, batch_size=args.test_batch_size,
                                     collate_fn=masksmiles.MaskSmilesDataset.collate,
                                     shuffle=False, **kwargs)

            return train_loader, val_loader, test_loader, vocab
