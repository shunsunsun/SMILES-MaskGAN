import os
import json
import torch

from maskgan.runs.maskgantrainer import MaskGANTrainer
from utils.args import maskgan_parser
from utils.mypath import save_path


def main(args):
 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    trainer = MaskGANTrainer(args, save_path)
    loss_history = trainer.run(args.epochs)

    with open(os.path.join(save_path, 'total_train_val_loss.json'), 'w') as train:
        json.dump(loss_history, train)


if __name__ == '__main__':
    parser = maskgan_parser()
    args = parser.parse_args()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, 'arg_parser.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    main(args)
