import os
import argparse

from utils import mypath


def ddpg_argparser():
    parser = argparse.ArgumentParser(description='Pytorch DDPG Training')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception'],
                        help='backbone name (default: xception)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='cabbage',
                        choices=['cabbage', 'cabbage5channel'],
                        help='dataset name (default: cabbage)')
    parser.add_argument('--input-channel', type=int, default='3',
                        help='input data size (default: 3)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')

    # training hyperparameters
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start-epoch', type=int, default=0,
                        metavar='N', help='start epochs (default: 0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=True,
                        help='whether to use balanced weights (default: False)')

    # optimizer parameters
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-2,
                        metavar='M', help='weight decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='whether use nesterov (default: False)')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

    # fine-tuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='fine-tuning on a different dataset')

    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    # choose model
    parser.add_argument('--models', type=str, default='DeepLab',
                        choices=['DeepLab', 'SegNet', 'UNet'],
                        help='Choose models (default: DeepLab)')

    return parser


def seq2seq_parser():
    parser = argparse.ArgumentParser(description='Pytorch SeqGAN Training')

    # dataloader
    parser.add_argument('--dataset', type=str, default='torchtext',
                        choices=['torchtext', 'Smiles'],
                        help='dataset')

    # Common parameters (Encoder, Attention, Decoder)
    parser.add_argument('--enc-hid-dim', type=int, default=100,
                        help='GRU cell hidden size (default: 200)')
    parser.add_argument('--dec-hid-dim', type=int, default=100,
                        help='GRU cell output size (default: 200)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='GRU cell output size (default: 0.2)')
    parser.add_argument('--bidirectional', type=bool, default=True,
                        help='bidirectional RNN cell (default: True)')

    # Encoder parameters
    parser.add_argument('--input-dim', type=int, default=None,
                        help='Vocab size')
    parser.add_argument('--enc-emb-dim', type=int, default=128,
                        help='Embedding layer out dim')

    # Decoder parameters
    parser.add_argument('--dec-emb-dim', type=int, default=64,
                        help='Embedding layer output dimension (default: auto)')
    parser.add_argument('--output-dim', type=int, default=None,
                        help='Embedding layer output dimension (default: auto)')

    # training hyperparameters
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start-epoch', type=int, default=0,
                        metavar='N', help='start epochs (default: 0)')
    parser.add_argument('--batch-size', type=int, default=1024,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--val-batch-size', type=int, default=1024,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for testing (default: auto)')

    # optimizer parameters
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-2,
                        metavar='M', help='weight decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='whether use nesterov (default: False)')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 123)')
    parser.add_argument('--cuda', type=str, default='cuda:0')

    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

    # fine-tuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='fine-tuning on a different dataset')

    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    return parser


def maskgan_parser():
    parser = argparse.ArgumentParser(description='Pytorch MaskGAN Training')

    # dataloader
    parser.add_argument('--dataset', type=str, default='masksmiles',
                        choices=['torchtext', 'Smiles', 'masksmiles'],
                        help='dataset')

    # mask probability
    parser.add_argument('--probability', type=float, default=0.3,
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
                        help='LSTM cell hidden size (default: 256)')
    parser.add_argument('--dec_hid_dim', type=int, default=128,
                        help='LSTM cell hidden size (default: 256)')
    parser.add_argument('--dec_out_emb_dim', type=int, default=128,
                        help='LSTM cell hidden size (default: 256)')
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
    parser.add_argument('--warmup_steps', type=int, default=5,
                        help='Cosine learning scheduler warmup steps (default: 10)')

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
