'''
Copyright (c) 2020 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
'''

from os.path import join, basename, dirname
import unittest

from seaqube.nlp.seaqube_model import SeaQuBeCompressLoader, BaseModelWrapper
from seaqube.nlp.tools import gensim_we_model_to_custom_we_model


class BaseFTGensimModel(BaseModelWrapper):
    def get_config(self):
        return dict(sg=self.model.sg, cbow_mean=self.model.cbow_mean, size=self.model.vector_size,
                    alpha=self.model.alpha, min_alpha=self.model.min_alpha, min_n=self.model.wv.min_n,
                    max_n=self.model.wv.max_n, window=self.model.window, min_count=self.model.vocabulary.min_count,
                    sample=self.model.vocabulary.sample, negative=self.model.negative, workers=self.model.workers,
                    epochs=self.define_epochs(), class_name=str(self))

    def _wrap_nlp_model(self, model):
        return gensim_we_model_to_custom_we_model(model)




class TestExampleBasicAugmentation(unittest.TestCase):
    def test_example001(self):
        # Import all Augmentation methods
        from seaqube.augmentation.word import Active2PassiveAugmentation, EDAAugmentation, TranslationAugmentation, EmbeddingAugmentation
        from seaqube.augmentation.char import QwertyAugmentation
        from seaqube.augmentation.corpus import UnigramAugmentation
        from seaqube.tools.io import load_json

        # prepare corpus and sample data
        text = 'The quick brown fox jumps over the lazy dog .'
        corpus = load_json(join(dirname(__file__), "..", "examples", "sick_full_corpus.json"))
        #print(corpus)

        # set up all augmentations

        # a (experimental) active to passive voice transformer. Only one sentences / doc to another
        a2p = Active2PassiveAugmentation()

        # easy-data augmentation method implementation (random word swap, insertion, deletion and replacement with synonyms)
        eda = EDAAugmentation(max_length=2)

        # translate text to other language and back (with Google Translater)
        translate = TranslationAugmentation(max_length=2)

        # replace words by a similiar one using another word embedding
        embed = EmbeddingAugmentation(max_length=2)

        # insert typos on text based on a qwerty-keyboard
        qwerty = QwertyAugmentation(replace_rate=0.07, max_length=2)

        # based on the UDA algorithm, only the Unigram method, which replace low meaning full words with other low meaning full words
        # this method needs a corpus, because it need to detect low meaningfull words
        unigram = UnigramAugmentation(corpus=corpus, max_length=2)



        ## API - Usage
        # Every augmentation object have the same possibility

        # 1. augmenting a string - same syntax as NLPAUG (https://github.com/makcedward/nlpaug)
        print(qwerty.augment(text))
        # or
        print(translate.augment(text))

        # 2. augmenting a doc (token based text)
        print(unigram.doc_augment(doc=corpus[0]))
        # doc_augment can also handle text:
        print(embed.doc_augment(text=text))

        # 3. augmenting a whole corpus
        print(eda(corpus[0:200]))

        # 4. Active2Passive is still experimental:
        a2p.doc_augment(doc=['someone', 'is', 'not', 'reading', 'the', 'email'])



        ## We want to apply a method on a corpus, train a model and measure the performance

        # tidy up RAM
        del unigram, embed, translate
        corpus_augmented = eda(corpus[0:200]) # augment a small subset

        # save on disk:
        #save_json(corpus_augmented, "augmented_sick.json")

        # To use NLP models which matching to or benchmark tool set, it must implement the 'BaseModelWrapper' interface.
        # We set up a class who implements the fasttext nlp model from the gensim package.
        # This is only needed to get the benchmark run

        from gensim.models import FastText

        class FTModelStd500V5(BaseFTGensimModel):
            def define_epochs(self):
                return 100

            def define_model(self):
                return FastText(sg=1, cbow_mean=1, size=300, alpha=0.025, min_alpha=0.0001, min_n=1, max_n=5,
                                window=5, min_count=1, sample=0.001, negative=5, workers=self.cpus - 1)

            def define_training(self):
                self.model.build_vocab(sentences=self.data, update=False)
                self.model.train(sentences=self.data, total_examples=len(self.data), epochs=self.epochs)

        model = FTModelStd500V5()

        # train the model
        # model.train_on_corpus(corpus_augmented)

        # get a dumped model to store it on disk - or use it in another process
        # model.get()
        # dill_dumper(model.get(), "example_model.dill")
        # or to save a compressed model:
        # SeaQuBeCompressLoader.save_model_compressed(model.get(), "example_model_compressed.dill")
        nlp = SeaQuBeCompressLoader.load_compressed_model(join(dirname(__file__), "..", "examples", "example_model_compressed.dill"), "example")

        del model

        from seaqube.benchmark.corpus4ir import Corpus4IRBenchmark
        from seaqube.benchmark.wordanalogy import WordAnalogyBenchmark
        from seaqube.benchmark.wordsimilarity import WordSimilarityBenchmark
        from seaqube.benchmark.wordoutliers import WordOutliersBenchmark

        wsb = WordSimilarityBenchmark(test_set='simlex999')
        print(wsb(nlp.model))  # score=0.008905456556563954

        wab = WordAnalogyBenchmark('google-analogies')
        print(wab(nlp.model))  # score=0.0

        wob = WordOutliersBenchmark('wikisem500')
        print(wob(nlp.model))  # score=0.0

        c4ir = Corpus4IRBenchmark(corpus[0:200])  # need the original corpus for setting up IR
        print(c4ir(nlp.model))


