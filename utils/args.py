import os
import argparse

from utils import mypath


def maskgan_parser():
    parser = argparse.ArgumentParser(description='Pytorch MaskGAN Training')

    # dataloader
    parser.add_argument('--dataset', type=str, default='masksmiles',
                        choices=['torchtext', 'masksmiles'],
                        help='dataset')

    # mask probability
    parser.add_argument('--probability', type=float, default=0.1,
                        help='Probability of mask ratio')

    # global network parameters
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout probability (default: 0.0)')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of lstm layers (default: 1)')
    parser.add_argument('--bidirectional', type=bool, default=True,
                        help='LSTM bidirectional option (default: True)')

    # LSTM base encoder parameters
    parser.add_argument('--enc_emb_dim', type=int, default=128,
                        help='LSTM cell hidden size (default: 128)')
    parser.add_argument('--enc_hid_dim', type=int, default=128,
                        help='LSTM cell hidden size (default: 128)')

    # LSTM base attention and decoder parameters
    parser.add_argument('--dec_emb_dim', type=int, default=128,
                        help='LSTM cell hidden size (default: 128)')
    parser.add_argument('--dec_hid_dim', type=int, default=128,
                        help='LSTM cell hidden size (default: 128)')
    parser.add_argument('--dec_out_emb_dim', type=int, default=128,
                        help='LSTM cell hidden size (default: 128)')
    parser.add_argument('--share_dec_input_output_emb', type=bool, default=False,
                        help='Share decoder input and output')

    # Seq2Seq model parameters

    # REINFORCE
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='Reward discount factor (default: 0.01)')
    parser.add_argument('--clip_value', type=float, default=5.0,
                        help='Clip value')

    # maskgan optimizer
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='MaskGAN optimizer learning rate (default: 1e-2)')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Optimizer weight decay (default: 1e-3)')
    parser.add_argument('--optim_clip_value', type=float, default=5.0,
                        help='gradient clip value (default: 5)')

    # learning scheduler
    parser.add_argument('--lr_gamma', type=float, default=0.9,
                        help='Exponential learning scheduler parameters (default: 0.5)')

    # cuda available
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='Using cuda (default: cuda:0)')

    # rollout
    parser.add_argument('--num_rollouts', type=int, default=1)

    # validation step
    parser.add_argument('--validate_every', type=int, default=1)

    # training hyperparameters
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start-epoch', type=int, default=0,
                        metavar='N', help='start epochs (default: 0)')
    parser.add_argument('--batch-size', type=int, default=512,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--val-batch-size', type=int, default=512,
                        metavar='N', help='input batch size for validation (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=100,
                        metavar='N', help='input batch size for testing (default: auto)')

    # save parameters
    parser.add_argument('--dist_file',
                        default=os.path.join(mypath.data_path, 'chembl26_canon_train.csv'),
                        help='Data path')
    parser.add_argument('--output_dir', default=mypath.save_path,
                        help='Save result path')
    parser.add_argument('--suite', default='v2')
    parser.add_argument('--vocab_load', default=None)
    parser.add_argument('--model_load', default=None)

    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 123)')

    return parser
