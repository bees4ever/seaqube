'''
Copyright (c) 2020 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
'''
import time
from os.path import join, basename, dirname
import unittest

from seaqube.benchmark.corpus4ir import Corpus4IRBenchmark
from seaqube.benchmark.semantic_wordnet import SemanticWordnetBenchmark
from seaqube.benchmark.wordanalogy import WordAnalogyBenchmark
from seaqube.benchmark.wordoutliers import WordOutliersBenchmark
from seaqube.benchmark.wordsimilarity import WordSimilarityBenchmark
from seaqube.nlp.seaqube_model import SeaQuBeNLPLoader, SeaQuBeCompressLoader
from seaqube.nlp.tools import tokenize_corpus
from seaqube.tools.io import load_json
from gensim.models import FastText
from seaqube.nlp.seaqube_model import BaseModelWrapper
from seaqube.nlp.tools import gensim_we_model_to_custom_we_model
from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.models.callbacks import CallbackAny2Vec
from progressbar import ProgressBar


class GensimEpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self, epochs):
        self.epoch = 0
        self.bar = ProgressBar(max_value=epochs)

    def on_train_begin(self, model: BaseWordEmbeddingsModel):
        pass

    def on_epoch_begin(self, model):
        pass

    def on_epoch_end(self, model):
        self.epoch += 1
        self.bar.update(self.epoch)


class BaseFTGensimModel(BaseModelWrapper):
    def get_config(self):
        return dict(sg=self.model.sg, cbow_mean=self.model.cbow_mean, size=self.model.vector_size,
                    alpha=self.model.alpha, min_alpha=self.model.min_alpha, min_n=self.model.wv.min_n,
                    max_n=self.model.wv.max_n, window=self.model.window, min_count=self.model.vocabulary.min_count,
                    sample=self.model.vocabulary.sample, negative=self.model.negative, workers=self.model.workers,
                    class_name=str(self))

    def _wrap_nlp_model(self, model):
        return gensim_we_model_to_custom_we_model(model)


class SmallFTModel(BaseFTGensimModel):
    def define_epochs(self):
        return 1000

    def define_model(self):
        return FastText(sg=1, cbow_mean=1, size=200, alpha=0.025, min_alpha=0.0001, min_n=1, max_n=6,
                        window=5, min_count=1, sample=0.001, negative=5, workers=self.cpus - 1,
                        callbacks=[GensimEpochLogger(self.epochs)])

    def define_training(self):
        self.model.build_vocab(sentences=self.data, update=False)
        self.model.train(sentences=self.data, total_examples=len(self.data), epochs=self.epochs)


## preparation, needs to be done not so often:
# SeaQuBeCompressLoader.save_model_compressed(model.get(), 'small_copressed_model.dill')


def load_corpus():
    return tokenize_corpus(load_json(join(dirname(__file__), 'test_data', 'small_corpus_01.json')))

def model_path():
    return join(dirname(__file__), 'test_data', 'small_copressed_model.dill')


class TestWordSimilarityBenchmark(unittest.TestCase):
    def test_simple_benchmark(self):
        # need to load a simple model, i.e. small dataset
        #model = SmallModel()
        #model.process(load_corpus())
        #nlp = SeaQuBeNLPLoader.load_model_from_tin_can(model.get(), 'small_model')
        nlp = SeaQuBeCompressLoader.load_compressed_model(model_path(), 'test_model')

        test_sets = ['semeval17', 'yp-130', 'mturk-771', 'verb-143', 'rg-65', 'simlex999', 'rw', 'simverb-3500', 'wordsim353-rel', 'men', 'mturk-287', 'mc-30', 'wordsim353-sim']
        scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.4959, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        print("vocab", nlp.model.vocabs())
        for i, test_set in enumerate(test_sets):
            simi_bench = WordSimilarityBenchmark(test_set)
            res = simi_bench(nlp.model)
            print(test_set, "result = ", res)
            self.assertAlmostEqual(res.score, scores[i], delta=0.1)


class TestWordAnalogyBenchmark(unittest.TestCase):
    def test_multiproc(self):
        nlp = SeaQuBeCompressLoader.load_compressed_model(model_path(), 'test_model')
        simi_bench = WordAnalogyBenchmark('google-analogies', max_cpus=10, multiprocessing=True)
        res = simi_bench(nlp.model)
        print("res = ", res)
        self.assertAlmostEqual(res.score, 0.0, delta=0.01)

    def test_nearest_neighbors(self):
        nlp = SeaQuBeCompressLoader.load_compressed_model(model_path(), 'test_model')

        test_set = 'google-analogies'
        mm = "NearestNeighbors"
        start = time.time()
        simi_bench = WordAnalogyBenchmark(test_set, method=mm, multiprocessing=False)
        res = simi_bench(nlp.model)
        end = time.time()
        print(test_set, "result = ", res, "time = ", end-start)
        self.assertAlmostEqual(res.score, 0.0, delta=0.01)

    def test_simple_benchmark(self):
        # need to load a simple model, i.e. small dataset
        #model = SmallModel()
        #model.process(load_corpus())
        #nlp = SeaQuBeNLPLoader.load_model_from_tin_can(model.get(), 'small_model')
        nlp = SeaQuBeCompressLoader.load_compressed_model(model_path(), 'test_model')

        for test_set in ['semeval', 'google-analogies', 'sat', 'msr', 'jair']:
            for mm in ['3CosAdd', 'VectorCalc', 'PairDir', 'SpaceEvolution', 'NearestNeighbors']:
                start = time.time()
                simi_bench = WordAnalogyBenchmark(test_set, method=mm, multiprocessing=True)
                res = simi_bench(nlp.model)
                end = time.time()
                print(test_set, "result = ", res, "time=", end-start)
                self.assertAlmostEqual(res.score, 0.0, delta=0.01)


class TestWordOutlierBenchmark(unittest.TestCase):
    def test_simple_benchmark(self):
        nlp = SeaQuBeCompressLoader.load_compressed_model(model_path(), 'test_model')

        for test_set in ['8-8-8', 'wikisem500']:
            simi_bench = WordOutliersBenchmark(test_set)
            res = simi_bench(nlp.model)
            print(test_set, "result = ", res)
            self.assertAlmostEqual(res.score, 0.0, delta=0.01)


class TestSemanticWordnet(unittest.TestCase):
    def test_with_wordnet(self):
        nlp = SeaQuBeCompressLoader.load_compressed_model(model_path(), 'test_model')
        vocabs = nlp.vocab()
        word_pairs, length = SemanticWordnetBenchmark.word_pairs_from_vocab_list(vocabs)
        print("Pairs Length:", length)

        self.assertEqual(length, 1378)  # this is predictable, except model/corpus changes

        swb = SemanticWordnetBenchmark(word_pairs, True, 2)
        self.assertEqual(length, 1378)
        results = swb(model=nlp.model).to_dict()
        self.assertEqual(type(results), dict)
        self.assertAlmostEquals(results["score"], 0.182, delta=0.2)  # this is not predictable, ...


class TestCorpus4IRBenchmark(unittest.TestCase):
    def test_simple_benchmark(self):
        # need to load a simple model, i.e. small dataset
        #model = SmallModel()
        #model.process(load_corpus())
        #nlp = SeaQuBeNLPLoader.load_model_from_tin_can(model.get(), 'small_model')
        nlp = SeaQuBeCompressLoader.load_compressed_model(model_path(), 'test_model')
        bench_corpus = Corpus4IRBenchmark(load_corpus())
        res = bench_corpus(nlp.model)

        self.assertAlmostEqual(res.score, 0.295066, delta=0.01)


if __name__ == "__main__":
    unittest.main()

