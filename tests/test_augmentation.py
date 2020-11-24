"""
Copyright (c) 2020 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""
import random
import time
import unittest
from itertools import product

import pytest

from seaqube.augmentation.base import AugmentationStreamer
from seaqube.augmentation.char.qwerty import QwertyAugmentation
from seaqube.augmentation.corpus.unigram import UnigramAugmentation
from seaqube.augmentation.reduction.unique_corpus import UniqueCorpusReduction
from seaqube.augmentation.word.active2passive import Active2PassiveAugmentation
from seaqube.augmentation.word.eda import EDAAugmentation
from seaqube.augmentation.word.embedding import EmbeddingAugmentation
from seaqube.augmentation.word.translation import TranslationAugmentation
from seaqube.nlp.tools import tokenize_corpus, unique_2d_list
from seaqube.tools.chainer import CallOnOneChain, ChainTerm
from seaqube.tools.math import lazy_sample

QUICK_FOX = "The quick brown fox jumps over the lazy dog"
QUICK_FOX_TOKENIZED = tokenize_corpus([QUICK_FOX])[0]

TEST_CORPUS = [['till', 'this', 'moment', 'i', 'never', 'knew', 'myself', '.'],
               ['seldom', ',', 'very', 'seldom', ',', 'does', 'complete', 'truth', 'belong', 'to', 'any', 'human',
                'disclosure', ';', 'seldom', 'can', 'it', 'happen', 'that', 'something', 'is', 'not', 'a', 'little',
                'disguised', 'or', 'a', 'little', 'mistaken', '.'],
               ['i', 'declare', 'after', 'all', 'there', 'is', 'no', 'enjoyment', 'like', 'reading', '!', 'how', 'much',
                'sooner', 'one', 'tires', 'of', 'anything', 'than', 'of', 'a', 'book', '!', '”'],
               ['men', 'have', 'had', 'every', 'advantage', 'of', 'us', 'in', 'telling', 'their', 'own', 'story', '.',
                'education', 'has', 'been', 'theirs', 'in', 'so', 'much', 'higher', 'a', 'degree'],
               ['i', 'wish', ',', 'as', 'well', 'as', 'everybody', 'else', ',', 'to', 'be', 'perfectly', 'happy', ';',
                'but', ',', 'like', 'everybody', 'else', ',', 'it', 'must', 'be', 'in', 'my', 'own', 'way', '.'],
               ['there', 'are', 'people', ',', 'who', 'the', 'more', 'you', 'do', 'for', 'them', ',', 'the', 'less',
                'they', 'will', 'do', 'for', 'themselves', '.'],
               ['one', 'half', 'of', 'the', 'world', 'can', 'not', 'understand', 'the', 'pleasures', 'of', 'the',
                'other', '.']]


class TestQwertyAugmentation(unittest.TestCase):
    def test_augmentation(self):
        qwerty = QwertyAugmentation(seed=42, replace_rate=0.09, max_length=3)
        result_one = [['the', 'quick', 'brown', 'fox', 'jumps', 'ove3', 'the', 'lazy', 'eog'], ['the', 'quick', 'brown', 'fod', 'jumps', '0ver', 'the', 'laxy', 'dog'], ['the', 'quick', 'nrown', 'fox', 'j6mps', 'over', 'the', 'lzzy', 'dog']]

        result_two = [['the', 'quici', 'brown', 'f9x', 'jumps', 'over', 'tve', 'lazy', 'dog'], ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'], ['the', 'quick', 'brown', 'fos', 'jUmps', 'over', 'the', '<wzy', 'doh']]

        result_three = [['the', 'qyick', 'brown', 'Fox', 'jumps', 'oveg', 'ths', 'lazy', 'dog'], ['the', 'quick', 'brodn', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'], ['tbe', 'quixk', 'brown', 'fox', 'jumps', 'ofeg', 'tye', 'lazy', 'sog']]


        self.assertEqual(qwerty.doc_augment(text=QUICK_FOX), result_one)
        self.assertEqual(qwerty.doc_augment(text=QUICK_FOX), result_two)
        self.assertEqual(qwerty.doc_augment(text=QUICK_FOX), result_three)

        qwerty = QwertyAugmentation(seed=42, replace_rate=0.09, max_length=3)
        self.assertEqual(qwerty.doc_augment(doc=QUICK_FOX_TOKENIZED), result_one)
        self.assertEqual(qwerty.doc_augment(doc=QUICK_FOX_TOKENIZED), result_two)
        self.assertEqual(qwerty.doc_augment(doc=QUICK_FOX_TOKENIZED), result_three)

    def test_call(self):
        qwerty = QwertyAugmentation(seed=42, replace_rate=0.09, max_length=3, remove_duplicates=True)
        self.assertEqual(len(qwerty(TEST_CORPUS)), 21)


class TestEDAAugmentation(unittest.TestCase):
    def test_augmentation(self):
        eda = EDAAugmentation(0.2, 0.3, 0.4, 0.07, 3, seed=42)
        print(eda.__doc__)

        result_one = [['the', 'quick', 'brown', 'slyboots', 'jumps', 'over', 'the', 'lazy', 'dog'],
                      ['the', 'quick', 'brown', 'fox',
                       'jumps', 'over', 'the', 'lazy',
                       'dog'], ['the', 'the', 'brown',
                                'dog', 'jumps', 'over',
                                'quick', 'lazy',
                                'fox'], ['the', 'quick',
                                         'brown', 'fox',
                                         'jumps',
                                         'over', 'the',
                                         'lazy', 'dog']]
        result_two = [['lazy', 'dog', 'brown', 'fox', 'over', 'jumps', 'the', 'the', 'quick'],
                      ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'hot', 'dog'],
                      ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy'],
                      ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']]

        augmentation_one = eda.doc_augment(text=QUICK_FOX)
        augmentation_two = eda.doc_augment(text=QUICK_FOX)

        # TODO get rid of the randomness in wordnet.synset, where a seed can not set, that's boring.
        self.assertEqual(len(augmentation_one), len(result_one))
        self.assertEqual(len(augmentation_two), len(result_two))

        eda = EDAAugmentation(0.2, 0.3, 0.4, 0.07, 3, seed=42)
        self.assertEqual(len(eda.doc_augment(doc=QUICK_FOX_TOKENIZED)), len(result_one))
        self.assertEqual(len(eda.doc_augment(doc=QUICK_FOX_TOKENIZED)), len(result_two))

    def test_call(self):
        eda = EDAAugmentation(seed=15, remove_duplicates=True, max_length=2)
        self.assertEqual(len(eda(TEST_CORPUS)), 14)


class TestEmbeddingAugmentation(unittest.TestCase):
    def test_embedding_augmentation(self):
        embeding = EmbeddingAugmentation(seed=42, max_length=10)
        result = [['the', 'quick', 'green', 'fox', 'spins', 'over', 'in', 'boring', 'dog'],
 ['the', 'easy', 'brown', 'fox', 'jump', 'over', 'which', 'dude', 'horse'],
 ['the', 'slow', 'green', 'fox', 'jumping', 'over', 'the', 'stupid', 'dog'],
 ['the', 'give', 'white', 'fox', 'jump', 'past', 'part', 'boring', 'dog'],
 ['the', 'give', 'gray', 'abc', 'jumps', 'over', 'the', 'lazy', 'cat'],
 ['which', 'quick', 'white', 'cbs', 'jump', 'past', 'the', 'dude', 'dogs'],
 ['which', 'give', 'brown', 'nbc', 'jump', 'over', 'part', 'lazy', 'dogs'],
 ['which', 'give', 'green', 'abc', 'jump', 'up', 'the', 'boring', 'cat'],
 ['part', 'quick', 'green', 'fox', 'spins', 'while', 'the', 'lazy', 'dogs'],
 ['in', 'easy', 'gray', 'fox', 'jumps', 'over', 'which', 'boring', 'dog']]
        augmented = embeding.doc_augment(text=QUICK_FOX)
        self.assertEqual(augmented, result)

        embeding = EmbeddingAugmentation(seed=42, max_length=10)
        self.assertEqual(embeding.doc_augment(doc=QUICK_FOX_TOKENIZED), result)

    def test_call(self):
        embeding = EmbeddingAugmentation(seed=42, remove_duplicates=True, max_length=2)
        self.assertEqual(len(embeding(TEST_CORPUS)), 14)

    def test_lazy_sample(self):
        r = random.Random()
        r.seed(123)
        P = product(range(10), repeat=100)
        L1 = lazy_sample(P, 10**100, 10, r)
        assert len(list(L1)) == 10


class TestUnigramAugmentation(unittest.TestCase):
    def test_call(self):
        unigram = UnigramAugmentation(corpus=TEST_CORPUS, seed=42, remove_duplicates=True, max_length=2)
        self.assertLessEqual(len(unigram(TEST_CORPUS)), 14)

    def test_unigram_augmentation(self):
        unigram = UnigramAugmentation(corpus=TEST_CORPUS, seed=42, replace_threshold=0.78, remove_duplicates=False, max_length=2, multiprocess=False)
        correct_result = [['till', 'this', 'moment', 'i', 'never', 'knew', 'myself', '.'], ['till', 'this', 'moment', 'the', 'never', 'knew', 'myself', '.'], ['seldom', ',', 'very', 'seldom', ',', 'does', 'complete', 'truth', 'belong', 'to', 'any', 'human', 'disclosure', ';', 'seldom', 'can', 'it', 'happen', 'that', 'something', 'is', 'not', 'a', 'little', 'disguised', 'or', 'a', 'little', 'mistaken', '.'], ['seldom', ',', 'very', 'seldom', ',', 'does', 'complete', 'truth', 'belong', 'to', 'any', 'human', 'disclosure', ';', 'seldom', 'can', 'it', 'happen', 'that', 'something', 'is', 'not', 'a', 'little', 'the', 'or', 'a', 'little', 'mistaken', '.'], ['i', 'declare', 'after', 'all', 'there', 'is', 'no', 'enjoyment', 'like', 'reading', '!', 'how', 'much', 'sooner', 'one', 'tires', 'of', 'anything', 'than', 'of', 'a', 'book', '!', '”'], [',', 'declare', 'after', 'all', 'there', 'is', 'no', 'enjoyment', 'like', 'reading', '!', 'how', 'much', 'sooner', 'one', 'tires', 'of', 'anything', 'than', 'of', 'a', 'book', '!', '”'], ['men', 'have', 'had', 'every', 'advantage', 'of', 'us', 'in', 'telling', 'their', 'own', 'story', '.', 'education', 'has', 'been', 'theirs', 'in', 'so', 'much', 'higher', 'a', 'degree'], ['men', 'have', 'had', 'every', 'advantage', 'the', 'us', 'in', 'telling', 'their', 'own', 'story', '.', 'education', 'has', 'been', 'theirs', 'in', 'so', 'much', 'higher', 'a', 'degree'], ['i', 'wish', ',', 'as', 'well', 'as', 'everybody', 'else', ',', 'to', 'be', 'perfectly', 'happy', ';', 'but', ',', 'like', 'everybody', 'else', ',', 'it', 'must', 'be', 'in', 'my', 'own', 'way', '.'], [',', 'wish', ',', 'as', 'well', 'as', 'everybody', 'else', ',', 'to', 'be', 'perfectly', 'happy', ';', 'but', ',', 'like', 'everybody', 'else', ',', 'it', 'must', 'be', 'in', 'my', 'own', 'way', '.'], ['there', 'are', 'people', ',', 'who', 'the', 'more', 'you', 'do', 'for', 'them', ',', 'the', 'less', 'they', 'will', 'do', 'for', 'themselves', '.'], ['seldom', 'are', 'people', ',', 'who', 'the', 'more', 'you', 'do', 'for', 'them', ',', 'the', 'less', 'they', 'will', 'do', 'for', 'themselves', '.'], ['one', 'half', 'of', 'the', 'world', 'can', 'not', 'understand', 'the', 'pleasures', 'of', 'the', 'other', '.'], ['the', 'half', 'of', 'the', 'world', 'can', 'not', 'understand', 'the', 'pleasures', 'of', 'the', 'other', '.']]

        self.assertEqual(unigram(TEST_CORPUS), correct_result)


class TestTranslationAugmentation(unittest.TestCase):
    def test_with_timeout(self):
        start = time.time()
        translation = TranslationAugmentation(timeout=0.5)

        translation.augment(QUICK_FOX)
        end = time.time()
        min_waiting = 0.5 * 9
        self.assertLess(min_waiting, end-start)

    

    def test_translation_augmentation(self):
        # a exact comparision is not easy possible, hence check some structure of the data
        translation = TranslationAugmentation()
        text_augment = translation.doc_augment(text=QUICK_FOX)
        self.assertGreater(len(text_augment), 0)
        self.assertGreater(len(text_augment[0]), 0)
        self.assertEqual(list, type(text_augment))
        self.assertEqual(list, type(text_augment[0]))


        translation = TranslationAugmentation()
        doc_augment = translation.doc_augment(doc=QUICK_FOX_TOKENIZED)
        print(doc_augment)

        self.assertGreater(len(doc_augment), 0)
        self.assertGreater(len(doc_augment[0]), 0)
        self.assertEqual(list, type(doc_augment))
        self.assertEqual(list, type(doc_augment[0]))



    def test_call(self):
        translation = TranslationAugmentation(max_length=10, remove_duplicates=True)
        augmented = translation(TEST_CORPUS)

        self.assertLessEqual(len(augmented), 7 * 10)


class TestActive2PassiveAugmentation(unittest.TestCase):
    def test_corpus(self):
        a2p = Active2PassiveAugmentation()
        assert a2p([['seldom', ',', 'very', 'seldom', ',', 'does', 'complete', 'truth', 'belong', 'to', 'any', 'human',
              'disclosure', ';', 'seldom', 'can', 'it', 'happen', 'that', 'something', 'is', 'not', 'a', 'little',
              'disguised', 'or', 'a', 'little', 'mistaken', '.']]) == [['Any', 'human', 'disclosure', 'has', 'been', 'completed', 'by', 'truth', 'seldom', ',', 'very', 'seldom', ',', 'belong', ';', 'seldom', 'can', 'it', 'happen', 'that', 'something', 'is', 'not', 'a', 'little', 'disguised', 'or', 'a', 'little', 'mistaken', '.']]

    def test_original_on(self):
        a2p = Active2PassiveAugmentation(original_too=True)
        assert a2p.augment("Simon reads a book. Jenny drinks tea.") == ['A book is read by Simon . Tea is drunk by Jenny.', 'simon reads a book . jenny drinks tea.']

    def test_exception(self):
        with pytest.raises(ValueError):
            a2p = Active2PassiveAugmentation()
            a2p.sentence2passive('He wants to drink') # This sentence has no object, so no transformation can be done

        ## this should not raise anything
        self.assertEqual(a2p([['He', 'wants', 'to', 'drink']]), [])



    def test_text(self):
        a2p = Active2PassiveAugmentation()

        def active2passive(text):
            return a2p.doc_augment(text=text)[0]

        assert active2passive("6.3-magnitude earthquake hits Taiwan.") == ['Taiwan', 'is', 'hit', 'by', 'earthquake', '6.3-magnitude', '.']

        assert active2passive("I was waiting for small Dina") == ['Small', 'Dina', 'was', 'being', 'waited', 'by', 'me']
        assert active2passive("I was waiting for this Diana") == ['This', 'Diana', 'was', 'being', 'waited', 'by', 'me']
        assert active2passive("I was waiting for Dina") == ['Dina', 'was', 'being', 'waited', 'by', 'me']
        assert active2passive("I was waiting for Dina. She is baking a cake.") == ['Dina', 'was', 'being', 'waited',
                                                                                   'by', 'me', '.', 'A', 'cake', 'is',
                                                                                   'being', 'baked', 'by', 'her', '.']
        assert active2passive("Simson cares the lion") == ['The', 'lion', 'is', 'cared', 'by', 'Simson']
        assert active2passive("Simson cares little Simon") == ['Little', 'Simon', 'is', 'cared', 'by', 'Simson']

        assert active2passive("They make cars in Detroit") == ['Cars', 'are', 'made', 'by', 'them', 'in', 'Detroit']
        assert active2passive("Mary cleans this room everyday") == ['This', 'room', 'is', 'cleaned', 'by', 'Mary',
                                                                    'everyday']
        assert active2passive("The water fills the tube") == ['The', 'tube', 'is', 'filled', 'by', 'the', 'water']
        assert active2passive("He knows me") == ['I', 'am', 'known', 'by', 'him']
        assert active2passive("He doesn't knows me") == ['I', 'am', 'not', 'known', 'by', 'him']

        assert active2passive("Ana does the homework") == ['The', 'homework', 'is', 'done', 'by', 'Ana']
        assert active2passive("They sell that house") == ['That', 'house', 'is', 'sold', 'by', 'them']
        assert active2passive("Jessica always plays the piano") == ['The', 'piano', 'is', 'always', 'played', 'by',
                                                                    'Jessica']
        assert active2passive("She buys a book") == ['A', 'book', 'is', 'bought', 'by', 'her']
        assert active2passive("Ratna is writing the letter") == ['The', 'letter', 'is', 'being', 'written', 'by',
                                                                 'Ratna']
        assert active2passive("She is doing her homework") == ['The', 'homework', 'is', 'being', 'done', 'by', 'her']
        assert active2passive("He is waiting for Dewi") == ['Dewi', 'is', 'being', 'waited', 'by', 'him']
        assert active2passive("Bobby is drawing a nice scenery") == ['A', 'nice', 'scenery', 'is', 'being', 'drawn',
                                                                     'by', 'Bobby']
        assert active2passive("They are giving the present") == ['The', 'present', 'is', 'being', 'given', 'by', 'them']
        assert active2passive("She cleaned the house") == ['The', 'house', 'was', 'cleaned', 'by', 'her']
        assert active2passive("Jeffri bought a new car") == ['A', 'new', 'car', 'was', 'bought', 'by', 'Jeffri']
        assert active2passive("The teacher called the students") == ['The', 'students', 'were', 'called', 'by', 'the',
                                                                     'teacher']
        assert active2passive("The teacher called the student") == ['The', 'student', 'was', 'called', 'by', 'the',
                                                                    'teacher']
        assert active2passive("She saved her money") == ['The', 'money', 'was', 'saved', 'by', 'her']
        assert active2passive("Rina paid all her purchases") == ['All', 'her', 'purchases', 'were', 'paid', 'by',
                                                                 'Rina']

        assert active2passive("he was playing a kite") == ['A', 'kite', 'was', 'being', 'played', 'by', 'him']
        assert active2passive("Andi was learning an English") == ['An', 'English', 'was', 'being', 'learned', 'by',
                                                                  'Andi']
        assert active2passive("They are building the house") == ['The', 'house', 'is', 'being', 'built', 'by', 'them']

        #  'PAST_PERFECT'
        assert active2passive("He had left the place") == ['The', 'place', 'had', 'been', 'left', 'by', 'him']
        assert active2passive("I had finished my work") == ['The', 'work', 'had', 'been', 'finished', 'by', 'me']
        assert active2passive("She had missed the last bus") == ['The', 'last', 'bus', 'had', 'been', 'missed', 'by',
                                                                 'her']
        assert active2passive("He had posted the latter") == ['The', 'latter', 'had', 'been', 'posted', 'by', 'him']
        assert active2passive("Rudi had not completed his speek") == ['The', 'speek', 'had', 'not', 'been', 'completed',
                                                                      'by', 'Rudi']

        # PRESENT_PERFECT
        assert active2passive("He has left the place") == ['The', 'place', 'has', 'been', 'left', 'by', 'him']
        assert active2passive("They have finally drunk the beer") == ['The', 'beer', 'has', 'finally', 'been', 'drunk',
                                                                      'by', 'them']
        assert active2passive("The cars have finished the race") == ['The', 'race', 'has', 'been', 'finished', 'by',
                                                                     'the', 'cars']

        # Present Perfect Progressive
        # assert active2passive("She has not been coming to Office since 12th July.") == ['Office', 'has', 'not', 'been', 'beeing', 'come', 'by', 'her', 'since', '12th', 'July', '.']
        # assert active2passive("He has been reading a newspaper for two hours.") == ['A', 'newspaper', 'has', 'been', 'beeing', 'read', 'by', 'him', 'for', 'two', 'hours', '.']
        # assert active2passive("They had been flying in a plane") == ['A', 'plane', 'had', 'been', 'beeing', 'flown', 'by', 'them']

        # FUTURE I SIMPLE WILL
        assert active2passive("I will read a newspaper") == ['A', 'newspaper', 'will', 'be', 'read', 'by', 'me']
        assert active2passive("Budi will not repair his bicycle") == ['The', 'bicycle', 'will', 'not', 'be', 'repaired',
                                                                      'by', 'Budi']
        assert active2passive("They will pay the tax") == ['The', 'tax', 'will', 'be', 'paid', 'by', 'them']

        # FUTURE I SIMPLE GOING TO
        assert active2passive("I am going to read a newspaper") == ['A', 'newspaper', 'is', 'going', 'to', 'be', 'read',
                                                                    'by', 'me']
        assert active2passive("The police are going to investigate the case.") == ['The', 'case', 'is', 'going', 'to',
                                                                                   'be', 'investigated', 'by', 'the',
                                                                                   'police', '.']
        # questions are not fully supported yet
        assert active2passive("Is anybody going to invite me?") == ['I', 'am', 'going', 'to', 'be', 'invited', 'by',
                                                                    'anybody', '?']

        # FUTURE II
        assert active2passive("They will have payed the tax") == ['The', 'tax', 'will', 'have', 'been', 'paid', 'by',
                                                                  'them']
        assert active2passive("The master will not have repaired his bicycle") == ['The', 'bicycle', 'will', 'not',
                                                                                   'have', 'been', 'repaired', 'by',
                                                                                   'the', 'master']

        # CONDITION I SIMPLE
        assert active2passive("He would eat the hamburger") == ['The', 'hamburger', 'would', 'be', 'eaten', 'by', 'him']
        # CONDITION II SIMPLE
        assert active2passive("He would have spoken the song") == ['The', 'song', 'would', 'have', 'been', 'spoken',
                                                                   'by', 'him']


class TestAugmentation(unittest.TestCase):
    def test_streaming(self):
        streamer = AugmentationStreamer([TranslationAugmentation(max_length=1), QwertyAugmentation(seed=424242, max_length=2)], reduction_chain=[UniqueCorpusReduction()])
        plain_output = streamer(TEST_CORPUS)

        self.assertAlmostEqual(len(plain_output), 102, delta=10)

    def test_chaining(self):
        import logging
        logging.basicConfig(level=logging.INFO)
        created_corpus = [['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']]

        pipe = CallOnOneChain([TranslationAugmentation(max_length=1), UnigramAugmentation(corpus=created_corpus, seed=50, max_length=2, replace_threshold=0.9, find_threshold=0.85),
                               QwertyAugmentation(seed=424242, max_length=2), unique_2d_list])

        augmented_and_unique = pipe(created_corpus)

        self.assertEqual(4, len(augmented_and_unique))


if __name__ == "__main__":
    unittest.main()
