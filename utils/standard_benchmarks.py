import pandas as pd

from rdkit import Chem

from utils.common_scoring_functions import TanimotoScoringFunction, RdkitScoringFunction, CNS_MPO_ScoringFunction, \
                                           IsomerScoringFunction, SMARTSScoringFunction
from utils.distribution_learning_benchmark import DistributionLearningBenchmark, NoveltyBenchmark, KLDivBenchmark
from utils.frechet_benchmark import FrechetBenchmark
from utils.score_modifier import MinGaussianModifier, MaxGaussianModifier, ClippedScoreModifier, GaussianModifier
from utils.scoring_function import ArithmeticMeanScoringFunction, GeometricMeanScoringFunction, ScoringFunction
from utils.descriptors import num_rotatable_bonds, num_aromatic_rings, logP, qed, tpsa, bertz, mol_weight, \
                              AtomCounter, num_rings


# Distribution-Learning-Benchmark
def novelty_benchmark(training_set_file: str, number_samples: int) -> DistributionLearningBenchmark:
    smiles_data = pd.read_csv(training_set_file)
    smiles_list = [s.strip() for s in smiles_data['canonical_smiles']]

    return NoveltyBenchmark(number_samples=number_samples, training_set=smiles_list)


def kldiv_benchmark(training_set_file: str, number_samples: int) -> DistributionLearningBenchmark:
    smiles_data = pd.read_csv(training_set_file)
    smiles_list = [s.strip() for s in smiles_data['canonical_smiles']]

    return KLDivBenchmark(number_samples=number_samples, training_set=smiles_list)


def frechet_benchmark(training_set_file: str, number_samples: int) -> DistributionLearningBenchmark:
    smiles_data = pd.read_csv(training_set_file)
    smiles_list = [s.strip() for s in smiles_data['canonical_smiles']]

    return FrechetBenchmark(training_set=smiles_list, sample_size=number_samples)


