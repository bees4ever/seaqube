"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""
import gc
from pandas import read_csv
import numpy as np
from progressbar import progressbar, ProgressBar
from sklearn.metrics import pairwise_distances
from seaqube.benchmark._benchmark import DataSetBasedWordEmbeddingBenchmark, get_shipped_test_set_path, \
    get_list_of_shipped_test_sets, BenchmarkScore
from seaqube.nlp.types import SeaQuBeWordEmbeddingsModel
from seaqube.package_config import log
from seaqube.tools.math import cosine
from seaqube.tools.umproc import ForEach


class WordAnalogyBenchmark(DataSetBasedWordEmbeddingBenchmark):
    """
    The very famous word embedding evaluation from Mikolov, Tomáš; Yih, Wen-tau und Zweig, Geoffrey
    (in Linguistic regularities in continuous space word representations) is here implemented with several algorothms
    available, where the NearestNeighbors is the fasted onel which however is copied from
    https://github.com/kudkudak/word-embeddings-benchmarks/blob/master/web/embedding.py
    """
    def __init__(self, test_set, method="3CosAdd", multiprocessing: bool = False, max_cpus=None):
        self.method = method
        self.max_cpus = max_cpus
        self.multiprocessing = multiprocessing
        super(WordAnalogyBenchmark, self).__init__(test_set)

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

    def _nearest_neighbors(self, a, b, c, model: SeaQuBeWordEmbeddingsModel, exclude=[]):

        """
        See https://github.com/kudkudak/word-embeddings-benchmarks/blob/master/web/embedding.py
        Find nearest neighbor of given word
        Parameters
        ----------
          word: string or vector
            Query word or vector.
          k: int, default: 1
            Number of nearest neighbours to return.
          metric: string, default: 'cosine'
            Metric to use.
          exclude: list, default: []
            Words to omit in answer
        Returns
        -------
          n: list
            Nearest neighbors.
        """
        k = 10
        metric = "cosine"

        v = b - a + c

        D = pairwise_distances(model.matrix(), v.reshape(1, -1), metric=metric)

        for w in exclude:
            D[self.word2index[w]] = D.max()

        return [(self.index2word[id], None) for id in D.argsort(axis=0).flatten()[0:k]]


    def _3_cos_add(self, a, b, c, model: SeaQuBeWordEmbeddingsModel, exclude=[]):
        vocab_len = model.matrix().shape[0]
        res = []
        for i in range(vocab_len):
            d = model.matrix()[i]
            res.append(cosine(d, c) - cosine(d, a) + cosine(d, b))

        sorted_zip = list(zip(
            np.array(model.vocabs())[np.argsort(res)[::-1]],  # [::-1] is for maximizing
            np.sort(res)[::-1]
        ))

        sorted_top = sorted_zip[0:10]
        del sorted_zip
        gc.collect()
        return sorted_top

    def _vector_calc(self, a, b, c, model: SeaQuBeWordEmbeddingsModel, exclude=[]):
        calculated_wv = a - b + c
        return self.most_similar(calculated_wv, model)

    def _pair_dir(self, a, b, c, model: SeaQuBeWordEmbeddingsModel, exclude=[]):
        vocab_len = model.matrix().shape[0]
        res = []
        for i in range(vocab_len):
            d = model.matrix()[i]
            res.append(cosine(d - c, b - a))

        return list(zip(
            np.array(model.vocabs())[np.argsort(res)[::-1]],  # [::-1] is for maximizing
            np.sort(res)[::-1]
        ))[0:10]

    def _space_evolution(self, a, b, c, model: SeaQuBeWordEmbeddingsModel, exclude=[]):
        a_h = a / np.linalg.norm(a)
        b_h = b / np.linalg.norm(b)
        c_h = c / np.linalg.norm(c)

        return self._3_cos_add(a_h, b_h, c_h, model)

    def apply_on_testset_line(self, row):
        a, b, c = row.word1.lower(), row.word2.lower(), row.word3.lower()
        log.info(f"WordAnalogy of these relation:{a}:{b}::{c}:?")
        target = row.target.lower()

        log.info(f"WordAnalogy: target={target}")

        detected_targets = self.measure_method(self.model.wv[a], self.model.wv[b],self.model.wv[c], self.model,
                                               exclude=[a, b, c])

        log.info(f"WordAnalogy: detected_targets={detected_targets}")

        word = detected_targets[0][0]

        del detected_targets
        log.info(f"WordAnalogy: compare: {word}=={target}?")
        return int(word == target)

    def __call__(self, model: SeaQuBeWordEmbeddingsModel) -> BenchmarkScore:
        correct_hits = 0

        if self.method == "3CosAdd":
            self.measure_method = self._3_cos_add
        elif self.method == 'VectorCalc':
            self.measure_method = self._vector_calc
        elif self.method == 'PairDir':
            self.measure_method = self._pair_dir
        elif self.method == 'SpaceEvolution':
            self.measure_method = self._space_evolution
        elif self.method == "NearestNeighbors":
            self.word2index = {word: i for i, word in enumerate(model.vocabs())}
            self.index2word = {i: word for i, word in enumerate(model.vocabs())}
            self.measure_method = self._nearest_neighbors
        else:
            raise ValueError(
                f"Argument `method` must be in one of [3CosAdd, VectorCalc, PairDir, SpaceEvolution, NearestNeighbors]")

        self.model = model

        # first filter dataset
        # all words need to be in the vocab list, otherwise it makes no sense

        model_vocabs = list(self.model.vocabs())

        def filter_vocabs(rowitem):
            _, row = rowitem
            if row.word1 in model_vocabs and row.word2 in model_vocabs and row.word3 in model_vocabs and row.target in model_vocabs:
                return row
            else:
                return None

        if self.multiprocessing:
            multi_wrapper = ForEach(filter_vocabs, max_cpus=self.max_cpus)
        else:
            def multi_wrapper(rows):
                for doc in rows:
                    yield filter_vocabs(doc)

        # for rowitem in progressbar(self.test_set.iterrows(), max_value=len(self.test_set)):
        #     _, row = rowitem
        #
        #     if row.word1 in self.model.vocabs() and row.word2 in self.model.vocabs() and row.word3 in self.model.vocabs() and row.target in self.model.vocabs():
        #         filtered_rows.append(row)

        filtered_rows = []
        prg = ProgressBar(max_value=len(self.test_set))
        for correct_flag in multi_wrapper(self.test_set.iterrows()):
            correct_hits += correct_flag
            prg.update(prg.value + 1)


        # first reduce model size -> save ram in copy process


        # then use filtered row for hard work
        prg = ProgressBar(max_value=len(filtered_rows))
        if self.multiprocessing:
            multi_wrapper = ForEach(self.apply_on_testset_line, max_cpus=self.max_cpus)
        else:
            def multi_wrapper(rows):
                for doc in rows:
                    yield self.apply_on_testset_line(doc)

        for correct_flag in multi_wrapper(filtered_rows):
            correct_hits += correct_flag
            prg.update(prg.value + 1)

        considered_lines = len(filtered_rows)

        if considered_lines == 0:
            return BenchmarkScore(0.0)

        return BenchmarkScore(correct_hits / considered_lines, {'matched_words': considered_lines,
                                                                'correct_hits': correct_hits})
