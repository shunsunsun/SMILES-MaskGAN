from typing import List

from utils.distribution_learning_benchmark import DistributionLearningBenchmark, ValidityBenchmark, \
                                                  UniquenessBenchmark
from utils.scoring_function import ArithmeticMeanScoringFunction
from utils.scoring_function import ArithmeticMeanScoringFunction
from utils.standard_benchmarks import novelty_benchmark, kldiv_benchmark, frechet_benchmark


def distribution_learning_benchmark_suite(chembl_file_path: str,
                                          version_name: str,
                                          number_samples: int) -> List[DistributionLearningBenchmark]:
    """
    Returns a suite of benchmarks for a specified benchmark version

    Args:
        chembl_file_path: path to CheMBL training set, necessary for some benchmarks
        version_name: benchmark version

    Returns:
        List of benchmarks
    """

    # For distribution-learning, v1 and v2 are identical
    if version_name == 'v1' or version_name == 'v2':

        return distribution_learning_suite_v1(chembl_file_path=chembl_file_path, number_samples=number_samples)

    raise Exception(f'Distribution-learning benchmark suite "{version_name}" does not exist.')


def distribution_learning_suite_v1(chembl_file_path: str, number_samples: int = 10000) -> \
                                   List[DistributionLearningBenchmark]:
    """
    Suite of distribution learning benchmarks, v1.

    Args:
        chembl_file_path: path to the file with the reference ChEMBL molecules

    Returns:
        List of benchmarks, version 1
    """

    return [
        ValidityBenchmark(number_samples=number_samples),
        UniquenessBenchmark(number_samples=number_samples),
        novelty_benchmark(training_set_file=chembl_file_path, number_samples=number_samples),
        kldiv_benchmark(training_set_file=chembl_file_path, number_samples=number_samples),
        frechet_benchmark(training_set_file=chembl_file_path, number_samples=number_samples)
    ]