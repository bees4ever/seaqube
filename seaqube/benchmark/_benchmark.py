'''
Copyright (c) 2020 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
'''

from os.path import join, exists

from pandas import DataFrame
from abc import abstractmethod
import os

from seaqube.nlp.types import SeaQuBeWordEmbeddingsModel
from seaqube.package_config import package_path
from seaqube.tools.types import Configable


class BenchmarkScore:
    def __init__(self, score, payload=None):
        self.score = score
        self.payload = payload

    def __str__(self):
        return f"(score={self.score}, payload={str(self.payload)})"

    def __repr__(self):
        return str(self)

class BaseWordEmbeddingBenchmark(Configable):
    @abstractmethod
    def method_name(self):
        """
        Return a human readable name of the benchmark method
        Returns str:
        """
        pass

    @abstractmethod
    def __call__(self, model: SeaQuBeWordEmbeddingsModel) -> BenchmarkScore:
        pass


class DataSetBasedWordEmbeddingBenchmark(BaseWordEmbeddingBenchmark):
    def __init__(self, test_set):
        if isinstance(test_set, str):
            self.test_set = self._load_shipped_test_set(test_set)
        elif isinstance(test_set, DataFrame):
            self.test_set = test_set
        else:
            raise ValueError("Please provide a shipped test set or an own DataFrame for Word Similarity Benchmark")

    @abstractmethod
    def _load_shipped_test_set(self, test_set):
        """
        load from package all the pre installed datasets - friendly cloned from https://github.com/vecto-ai/word-benchmarks
        Args:
            test_set: name of test_set
        Returns: csv as DataFrame
        """
        pass

    def get_config(self):
        return dict(class_name=str(self), test_set=self.test_set)


shipped_datasets = {
        'word-similarity': join(package_path, "benchmark", "benchmark_datasets", "word-similarity", "monolingual", "en"),
        'word-analogy': join(package_path, "benchmark", "benchmark_datasets", "word-analogy", "monolingual", "en"),
    }


def get_shipped_test_set_path(category, name):
    base_path = shipped_datasets[category]

    path = join(base_path, name + ".csv")

    print("DEBUGGGG: LOAD=", path)
    if not exists(path):
        raise KeyError(f"Provided test set with name={name} does not exist")

    return path


def get_list_of_shipped_test_sets(category):
    base_path = shipped_datasets[category]

    return list(map(lambda x: x.replace(".csv", ""), os.listdir(base_path)))



