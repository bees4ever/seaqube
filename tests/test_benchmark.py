'''
Copyright (c) 2020 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
'''

from os.path import join, basename, dirname
import unittest

from seaqube.benchmark.corpus4ir import Corpus4IRBenchmark
from seaqube.benchmark.wordanalogy import WordAnalogyBenchmark
from seaqube.benchmark.wordsimilarity import WordSimilarityBenchmark
from seaqube.nlp.seaqube_model import SeaQuBeNLPLoader
from seaqube.nlp.tools import tokenize_corpus
from seaqube.tools.io import load_json
from SeaQuBeRepo.tests.test_data.small_model import SmallModel


def load_corpus():
    return tokenize_corpus(load_json(join(dirname(__file__), 'test_data', 'small_corpus_01.json')))


class TestWordSimilarityBenchmark(unittest.TestCase):
    def test_simple_benchmark(self):
        # need to load a simple model, i.e. small dataset
        model = SmallModel()
        model.process(load_corpus())
        nlp = SeaQuBeNLPLoader.load_model_from_tin_can(model.get(), 'small_model')

        test_sets = ['semeval17', 'yp-130', 'mturk-771', 'verb-143', 'rg-65', 'simlex999', 'rw', 'simverb-3500', 'wordsim353-rel', 'men', 'mturk-287', 'mc-30', 'wordsim353-sim']
        scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.23, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        print("vocab", nlp.model.vocabs())
        for i, test_set in enumerate(test_sets):
            simi_bench = WordSimilarityBenchmark(test_set)
            res = simi_bench(nlp.model)
            print(test_set, "result = ", res)
            self.assertAlmostEqual(res.score, scores[i], delta=0.1)


class TestWordAnalogyBenchmark(unittest.TestCase):
    def test_simple_benchmark(self):
        # need to load a simple model, i.e. small dataset
        model = SmallModel()
        model.process(load_corpus())
        nlp = SeaQuBeNLPLoader.load_model_from_tin_can(model.get(), 'small_model')

        for test_set in ['semeval', 'google-analogies', 'sat', 'msr', 'jair']:
            simi_bench = WordAnalogyBenchmark(test_set)
            res = simi_bench(nlp.model)
            print(test_set, "result = ", res)
            self.assertAlmostEqual(res.score, 0.0, delta=0.01)


class TestCorpus4IRBenchmark(unittest.TestCase):
    def test_simple_benchmark(self):
        # need to load a simple model, i.e. small dataset
        model = SmallModel()
        model.process(load_corpus())
        nlp = SeaQuBeNLPLoader.load_model_from_tin_can(model.get(), 'small_model')
        bench_corpus = Corpus4IRBenchmark(load_corpus())
        res = bench_corpus(nlp.model)

        self.assertAlmostEqual(res.score, 0.5760, delta=0.01)


if __name__ == "__main__":
    unittest.main()

