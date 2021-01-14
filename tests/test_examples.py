"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

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

    def test_example_aug(self):
        # Import all Augmentation methods
        from seaqube.augmentation.word import Active2PassiveAugmentation, EDAAugmentation, TranslationAugmentation, \
            EmbeddingAugmentation
        from seaqube.augmentation.char import QwertyAugmentation
        from seaqube.augmentation.corpus import UnigramAugmentation

        # import some tools
        from seaqube.tools.io import load_json
        from os.path import join

        # load example data
        import json, urllib.request
        data = urllib.request.urlopen(
            "https://raw.githubusercontent.com/bees4ever/SeaQuBe/master/examples/sick_full_corpus.json").read()

        corpus = json.loads(data)
        text = 'The quick brown fox jumps over the lazy dog .'

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

    def test_example_chain(self):
        from seaqube.augmentation.char.qwerty import QwertyAugmentation
        from seaqube.augmentation.corpus.unigram import UnigramAugmentation
        from seaqube.augmentation.word.active2passive import Active2PassiveAugmentation
        from seaqube.augmentation.word.eda import EDAAugmentation
        from seaqube.augmentation.word.embedding import EmbeddingAugmentation
        from seaqube.augmentation.word.translation import TranslationAugmentation

        TEST_CORPUS = [['till', 'this', 'moment', 'i', 'never', 'knew', 'myself', '.'],
                       ['seldom', ',', 'very', 'seldom', ',', 'does', 'complete', 'truth', 'belong', 'to', 'any',
                        'human',
                        'disclosure', ';', 'seldom', 'can', 'it', 'happen', 'that', 'something', 'is', 'not', 'a',
                        'little',
                        'disguised', 'or', 'a', 'little', 'mistaken', '.'],
                       ['i', 'declare', 'after', 'all', 'there', 'is', 'no', 'enjoyment', 'like', 'reading', '!', 'how',
                        'much',
                        'sooner', 'one', 'tires', 'of', 'anything', 'than', 'of', 'a', 'book', '!', '”'],
                       ['men', 'have', 'had', 'every', 'advantage', 'of', 'us', 'in', 'telling', 'their', 'own',
                        'story', '.',
                        'education', 'has', 'been', 'theirs', 'in', 'so', 'much', 'higher', 'a', 'degree'],
                       ['i', 'wish', ',', 'as', 'well', 'as', 'everybody', 'else', ',', 'to', 'be', 'perfectly',
                        'happy', ';',
                        'but', ',', 'like', 'everybody', 'else', ',', 'it', 'must', 'be', 'in', 'my', 'own', 'way',
                        '.'],
                       ['there', 'are', 'people', ',', 'who', 'the', 'more', 'you', 'do', 'for', 'them', ',', 'the',
                        'less',
                        'they', 'will', 'do', 'for', 'themselves', '.'],
                       ['one', 'half', 'of', 'the', 'world', 'can', 'not', 'understand', 'the', 'pleasures', 'of',
                        'the',
                        'other', '.']]

        from seaqube.augmentation.base import AugmentationStreamer
        from seaqube.augmentation.reduction.unique_corpus import UniqueCorpusReduction
        # Here we set up a augmentation stream. Every document will passed trought this augmentation line by line.
        # This means: a document _d_ will be in the first step translated.
        # In the second step, this translated document is feed to Qwerty. Now, qwerty returns 2 documents.
        # This 2 documents will be each feed to EDA. EDA geenrates 4 augmented documents for the two inputs, i.e. one line results in 8 lines output.
        # AugmentationStreamer can also reduce documents, here it reduce it, using the unique reducer.
        streamer = AugmentationStreamer(
            [TranslationAugmentation(max_length=1), QwertyAugmentation(seed=424242, max_length=2),
             EDAAugmentation(max_length=4)], reduction_chain=[UniqueCorpusReduction()])

        augmented_doc = streamer([TEST_CORPUS[0]])
        augmented_doc
        len(augmented_doc)  # after reducing documents can be less then 8

        streamer(TEST_CORPUS)  # apply the full corpus for the streamer

        corpus = [['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']]

        from seaqube.tools.chainer import CallOnOneChain
        from seaqube.nlp.tools import unique_2d_list

        pipe = CallOnOneChain([TranslationAugmentation(max_length=1),
                               UnigramAugmentation(corpus=TEST_CORPUS, seed=50, max_length=2, replace_threshold=0.9,
                                                   find_threshold=0.85),
                               QwertyAugmentation(seed=424242, max_length=2), unique_2d_list])

        augmented_and_unique = pipe(TEST_CORPUS)

        augmented_and_unique

        len(augmented_and_unique)  # 8 * 2 * 2 = 32, reducing makes it smaller



    def test_example_nlp(self):
        from seaqube.nlp.types import SeaQuBeWordEmbeddingsModel
        # Lets have a look at a contexted based NLP model, called Context2Vec
        from seaqube.nlp.context2vec.context2vec import Context2Vec

        # Import some seaqube tools:
        from seaqube.nlp.tools import word_count_list
        from seaqube.nlp.types import RawModelTinCan
        from seaqube.nlp.seaqube_model import SeaQuBeNLPLoader, SeaQuBeCompressLoader
        from seaqube.nlp.tools import tokenize_corpus
        class SeaQuBeWordEmbeddingsModelC2V(SeaQuBeWordEmbeddingsModel):
            def __init__(self, c2v: Context2Vec):
                self.c2v = c2v

            def vocabs(self):
                return self.c2v.wv.vocabs

            @property
            def wv(self):
                return self.c2v.wv

            def word_vector(self, word):
                return self.c2v.wv[word]

            def matrix(self):
                return self.c2v.wv.matrix

        star_wars_cites = ["How you get so big eating food of this kind?", "'Spring the trap!'", "Same as always…",
                           "You came in that thing? You’re braver than I thought!", "Who’s scruffy looking?",
                           "Let the Wookiee win.", "The Emperor is not as forgiving as I am",
                           "I don’t know where you get your delusions, laserbrain.", "Shutting up, sir,",
                           "Boring conversation anyway…", ]
        corpus = tokenize_corpus(star_wars_cites)
        c2v = Context2Vec(epoch=3)
        c2v.train(corpus)

        seaC2V = SeaQuBeWordEmbeddingsModelC2V(c2v)

        tin_can = RawModelTinCan(seaC2V, word_count_list(corpus))

        SeaQuBeCompressLoader.save_model_compressed(tin_can, "c2v_small")

        nlp = SeaQuBeNLPLoader.load_model_from_tin_can(tin_can, "c2v")

        nlp("This is a test")

        doc = list(nlp("This is a test"));
        print(doc);
        type(doc[0])
        nlp("This is a test")[0].vector

        nlp("This is a test").vector
        nlp("This is a test").sif_vector

        nlp("Is the Emperor a laserbrain").similarity("Boring conversation anyway…")

        nlp("Is the Emperor a laserbrain").similarity("Boring conversation anyway…", vector="sif")
        word = nlp("Is the Emperor a laserbrain?")[2]

        word
        word.similarity("Wookiee")
        nlp.vocab()



    def test_example_benchmark(self):
        # import some tools
        from seaqube.tools.io import load_json
        from os.path import join
        from seaqube.nlp.seaqube_model import SeaQuBeCompressLoader
        main_path = join(dirname(__file__), "..", "examples")
        nlp = SeaQuBeCompressLoader.load_compressed_model(join(main_path, "example_model_compressed.dill"), "example")

        import json, urllib.request
        data = urllib.request.urlopen(
            "https://raw.githubusercontent.com/bees4ever/SeaQuBe/master/examples/sick_full_corpus.json").read()

        corpus = json.loads(data)

        ## import tools
        from seaqube.benchmark.wordanalogy import WordAnalogyBenchmark
        from seaqube.benchmark.wordsimilarity import WordSimilarityBenchmark
        from seaqube.benchmark.wordoutliers import WordOutliersBenchmark
        from seaqube.benchmark.semantic_wordnet import SemanticWordnetBenchmark

        # We need to install `vec4ir`, this can be done trough "SeaQuBe":
        # from seaqube.benchmark.corpus4ir import Corpus4IRBenchmark
        from seaqube import download;
        download('vec4ir')

        import vec4ir
        # load module
        from seaqube.benchmark.corpus4ir import Corpus4IRBenchmark

        # perform semantical tests
        wsb = WordSimilarityBenchmark(test_set='simlex999')
        print(wsb(nlp.model))  # score=0.008905456556563954

        wab = WordAnalogyBenchmark('google-analogies')
        print(wab(nlp.model))  # score=0.0

        wob = WordOutliersBenchmark('wikisem500')
        print(wob(nlp.model))  # score=0.0

        c4ir = Corpus4IRBenchmark(corpus[0:200])  # need the original corpus for setting up IR
        print(c4ir(nlp.model))

        # The semantic word net benchmark needs a base of word pairs. This pairs can be generated easily:
        vocabs = nlp.vocab()
        vocabs = vocabs[0:200]

        word_pairs, length = SemanticWordnetBenchmark.word_pairs_from_vocab_list(vocabs)
        print("Pairs Length:", length)

        swb = SemanticWordnetBenchmark(word_pairs, False)
        print(swb(nlp.model))




