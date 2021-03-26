import os
import tqdm
import random
import torch
import numpy as np

from typing import List
from collections import namedtuple, defaultdict, OrderedDict

from utils.assess_distribution_learning import assess_distribution_learning
from utils.distribution_matching_generator import DistributionMatchingGenerator
from utils.helpers import setup_default_logger
from utils.args import maskgan_parser
from utils import mypath

from maskgan.maskgan_networks import MaskGANGenerator
from dataloaders import make_data_loader

from baselines.moses.script_utils import set_seed


class MaskGANTestGenerator(DistributionMatchingGenerator):
    def __init__(self, args):
        self.args = args
        self.encode_dict = {"Br": 'Y', "Cl": 'X', "Si": 'A', 'Se': 'Z', '@@': 'R', 'se': 'E'}
        self.decode_dict = {v: k for k, v in self.encode_dict.items()}

        # Define test Dataloader & load saved vocab
        kwargs = {'pin_memory': False, 'num_workers': 8}
        _, _, self.test_loader, self.vocab = make_data_loader(args, **kwargs, mode='test')

        test_sample = [sample for i, sample in enumerate(self.test_loader)]

        self.sample_data = random.sample(test_sample, 100)

        # make id to character
        self.id2char = {v: k for k, v in self.vocab.items()}

        # Define task
        Task = namedtuple('Task', 'source_dictionary target_dictionary')
        task = Task(source_dictionary=self.vocab,
                    target_dictionary=self.vocab)

        # Define saved model
        if args.model_load is None:
            args.model_load = os.path.join(mypath.save_path, '50_maskgan.pt')
        
        model_state = torch.load(args.model_load)

        generator_state = OrderedDict()
        for k, v in model_state.items():
            if 'generator' in k:
                new_k = k.replace('generator.', '')
                generator_state[new_k] = v

        self.model = MaskGANGenerator(args, task)
        self.model.load_state_dict(generator_state)
        self.model = self.model.to(args.cuda)
        self.model.eval()

    def id2string(self, ids):

        string = ''.join([self.id2char[id] for id in ids])

        return string

    def tensor2string(self, target, tensor):

        ids_ = tensor.tolist()
        ids = [id for id in ids_ if id != 0 and id != 1 and id != 2 and id != 47]

        tgt_ids_ = target.tolist()
        tgt_ids = [id for id in tgt_ids_ if id != 0 and id != 1 and id != 2 and id != 47]

        new_ids = []
        for id, tgt_id in zip(ids, tgt_ids):

            if np.random.uniform() < 0.01:
                new_ids.append(tgt_id)

            else:
                new_ids.append(id)

        string = self.id2string(new_ids)

        return self.decode(string)

    def decode(self, smiles):
        """
        Replace special tokens with their multi-character equivalents.

        Args:
            smiles: SMILES string
        
        Returns:
            SMILES string with possibly multi-char tokens
        """

        temp_smiles = smiles
        for symbol, token in self.decode_dict.items():
            temp_smiles = temp_smiles.replace(symbol, token)
        
        return temp_smiles

    def sample(self, data_samples):
        """
        n_batch: test batch size
        data_samples: test data samples
        """

        samples = [sample.to(self.args.cuda) for sample in data_samples]
        srcs, tgts, lengths, masks = samples

        # outputs, logits, _ = self.model.forward(srcs, lengths, tgts, masks)
        logits = self.model.logits(srcs, lengths, tgts, masks)
        outputs = torch.argmax(logits, dim=-1)

        return [self.tensor2string(tgt, tensor)
                for tgt, tensor in zip(tgts, outputs)]

    def generate(self, number_samples: int) -> List[str]:
        samples = []
        n = number_samples

        with tqdm.tqdm(total=number_samples, desc='Generating samples') as T:

            while n > 0:
                for sample in self.sample_data:
                    current_samples = self.sample(sample)
                    samples.extend(current_samples)

                    n -= len(current_samples)
                    T.update(len(current_samples))

        return samples


def main(args):

    setup_default_logger()
    set_seed(args.seed)

    generator = MaskGANTestGenerator(args)

    json_file_path = os.path.join(args.output_dir, f'distribution_learning_results_{args.probabiltiy}.json')
    assess_distribution_learning(generator,
                                 chembl_training_file=args.dist_file,
                                 json_output_file=json_file_path,
                                 benchmark_version=args.suite)


if __name__ == '__main__':
    parser = maskgan_parser()
    args = parser.parse_args()

    main(args)
