"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""


import re
from copy import deepcopy
import numpy as np
from pandas import read_csv
from progressbar import progressbar

from seaqube.benchmark._benchmark import DataSetBasedWordEmbeddingBenchmark, get_shipped_test_set_path, \
    get_list_of_shipped_test_sets, BenchmarkScore
from seaqube.nlp.types import SeaQuBeWordEmbeddingsModel


class NotInVocabularyError(Exception):
    """
    Wraps a not-in-vocab error
    """
    pass


class WordOutliersBenchmark(DataSetBasedWordEmbeddingBenchmark):
    """
    The word outlier method implemented based on public available dataset, which requires a pre processing for
    creating word outlier challenges.

    The idea is simply to detect a outlier word of a group of words.

    For example `car` is the outlier in `["apple","pear","car","banana","orange"]`
    """
    def method_name(self):
        return "wordoutliers"

    def _load_shipped_test_set(self, test_set):
        """"

        """
        path = get_shipped_test_set_path("word-outliers", test_set)
        return read_csv(path)

    def available_test_sets(self):
        return get_list_of_shipped_test_sets("word-outliers")

    def __call__(self, model: SeaQuBeWordEmbeddingsModel) -> BenchmarkScore:

        def load(s):
            chunks = re.split(',\ ', s.strip()[1:-1])
            return [re.sub(r"['\"]$", '', re.sub(r"^['\"]+", '', ch)).strip() for ch in chunks]


        test_run = 0
        test_succeeded = 0

        for rowitem in progressbar(self.test_set.iterrows(), max_value=len(self.test_set)):
            _, row = rowitem

            words: list = load(row.words)
            outliers: list = load(row.outliers)

            for outlier in outliers:
                if len(outlier) > 0:
                    test_set = deepcopy(words)
                    test_set.append(outlier)
                    try:
                        test_succeeded += int(self._outlier_detection_test(test_set, outlier, model))
                        test_run += 1
                    except NotInVocabularyError:
                        pass

        if test_run == 0:
            return BenchmarkScore(0.0)

        return BenchmarkScore(test_succeeded/test_run, {'test_run': test_run, 'test_succeeded': test_succeeded})


    def _outlier_detection_test(self, words, outlier, model: SeaQuBeWordEmbeddingsModel) -> bool:
        outlier_index = words.index(outlier)
        if all([word in model.vocabs() for word in words]):
            dim = len(words)
            sim_matrix = np.zeros(dim * dim).reshape([dim, dim])
            for i, first_word in enumerate(words):
                for j, second_word in enumerate(words):
                    sim_matrix[i][j] = model.similarity(first_word, second_word)

            #print(sim_matrix)
            word_sum = np.sum(sim_matrix, axis=1)
            #print(word_sum)
            return np.argmin(word_sum) == outlier_index

        else:
            raise NotInVocabularyError("Wordlist could not used, because words are missing")

