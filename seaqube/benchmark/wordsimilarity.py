"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

from pandas import read_csv
from progressbar import progressbar
from seaqube.benchmark._benchmark import DataSetBasedWordEmbeddingBenchmark, get_shipped_test_set_path, \
    get_list_of_shipped_test_sets, BenchmarkScore, complete_correlation_calculation
from seaqube.nlp.types import SeaQuBeWordEmbeddingsModel
from seaqube.package_config import log


class WordSimilarityBenchmark(DataSetBasedWordEmbeddingBenchmark):
    """
    This class implements the word similarity challenge which is based on human scored word pairs.

    See also Milajevs, Dmitrijs und Griffiths, Sascha: A proposal for linguistic similarity datasets based on
            commonality lists. arXiv preprint arXiv:1605.04553 (2016)


    """
    def method_name(self):
        return "wordsimilarity"

    def _load_shipped_test_set(self, test_set):
        """"

        """
        path = get_shipped_test_set_path("word-similarity", test_set)
        return read_csv(path)

    def available_test_sets(self):
        return get_list_of_shipped_test_sets("word-similarity")

    def __call__(self, model: SeaQuBeWordEmbeddingsModel) -> BenchmarkScore:
        model_sim, sheet_sim = [], []
        for rowitem in progressbar(self.test_set.iterrows(), max_value=len(self.test_set)):
            _, row = rowitem

            if row.word1 in model.vocabs() and row.word2 in model.vocabs():
                model_sim.append(model.similarity(row.word1, row.word2))
                sheet_sim.append(row.similarity)
        log.info(
            f"{self.__class__.__name__}: Correlation of list of length={len(model_sim)}")
        if len(model_sim) < 2:
            return BenchmarkScore(0.0)

        return complete_correlation_calculation(model_sim, sheet_sim)
