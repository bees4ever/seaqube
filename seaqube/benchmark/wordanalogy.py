'''
Copyright (c) 2020 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
'''

from pandas import read_csv
import numpy as np
from progressbar import progressbar

from seaqube.benchmark._benchmark import DataSetBasedWordEmbeddingBenchmark, get_shipped_test_set_path, \
    get_list_of_shipped_test_sets, BenchmarkScore
from seaqube.nlp.types import SeaQuBeWordEmbeddingsModel
from seaqube.package_config import log
from seaqube.tools.math import cosine


class WordAnalogyBenchmark(DataSetBasedWordEmbeddingBenchmark):
    def method_name(self):
        return "wordanalogy"

    def _load_shipped_test_set(self, test_set):
        """"

        """
        path = get_shipped_test_set_path("word-analogy", test_set)
        return read_csv(path)

    def available_test_sets(self):
        return get_list_of_shipped_test_sets("word-analogy")

    def most_similar(self, calculated_vector, model: SeaQuBeWordEmbeddingsModel, topn=10):
        vocab_len = len(model.vocabs())

        distances = np.array([np.linalg.norm(calculated_vector - model.matrix()[i]) for i in range(vocab_len)])
        found_indecies = np.array(distances).argsort()[0:topn]

        return list(zip(np.array(model.vocabs())[found_indecies], distances[found_indecies]))

    def _3_cos_add(self, a, b, c, model: SeaQuBeWordEmbeddingsModel):
        vocab_len = model.matrix().shape[0]
        res = []
        for i in range(vocab_len):
            d = model.matrix()[i]
            res.append(cosine(d, c) - cosine(d, a) + cosine(d, b))


        return list(zip(
                    np.array(model.vocabs())[np.argsort(res)[::-1]], # [::-1] is for maximizing
                    np.sort(res)[::-1]
                ))[0:10]

    def _vector_calc(self, a, b, c, model: SeaQuBeWordEmbeddingsModel):
        calculated_wv = a - b + c
        return self.most_similar(calculated_wv, model)

    def _pair_dir(self, a, b, c, model: SeaQuBeWordEmbeddingsModel):
        vocab_len = model.matrix().shape[0]
        res = []
        for i in range(vocab_len):
            d = model.matrix()[i]
            res.append(cosine(d - c, b - a))

        return list(zip(
            np.array(model.vocabs())[np.argsort(res)[::-1]],  # [::-1] is for maximizing
            np.sort(res)[::-1]
        ))[0:10]

    def _space_evolution(self, a, b, c, model: SeaQuBeWordEmbeddingsModel):
        a_h = a / np.linalg.norm(a)
        b_h = b / np.linalg.norm(b)
        c_h = c / np.linalg.norm(c)

        return self._3_cos_add(a_h, b_h, c_h, model)

    def __call__(self, model: SeaQuBeWordEmbeddingsModel, method="3CosAdd") -> BenchmarkScore:
        considered_lines = 0
        correct_hits = 0

        if method == "3CosAdd":
            measure_method = self._3_cos_add
        elif method == 'VectorCalc':
            measure_method = self._vector_calc
        elif method == 'PairDir':
            measure_method = self._pair_dir
        elif method == 'SpaceEvolution':
            measure_method = self._space_evolution
        else:
            raise ValueError(f"Argument `method` must be in one of [3CosAdd, VectorCalc, PairDir, SpaceEvolution]")

        for rowitem in progressbar(self.test_set.iterrows()):
            _, row = rowitem
            if row.word1 in model.vocabs() and row.word2 in model.vocabs() and row.word3 in model.vocabs() and row.target in model.vocabs():
                considered_lines += 1  # all words need to be in the vocab list, otherwise it makes no sense

                log.debug(f"WordAnalogy of these relation:{row.word1}:{row.word2}::{row.word3}:?")
                print(f"WordAnalogy of these relation:{row.word1}:{row.word2}::{row.word3}:?")

                log.debug(f"WordAnalogy: target={row.target}")
                print("target", row.target)
                detected_outliers = measure_method(model.wv[row.word1], model.wv[row.word2], model.wv[row.word3], model)

                log.debug(f"WordAnalogy: detected_outliers={detected_outliers}")
                print(f"WordAnalogy: detected_outliers={detected_outliers}")

                word = detected_outliers[0][0]
                correct_hits += int(word == row.target)

        if considered_lines == 0:
            return BenchmarkScore(0.0)
            
        return BenchmarkScore(correct_hits / considered_lines, {'matched_words': considered_lines})
