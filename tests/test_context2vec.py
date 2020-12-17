import unittest
from os.path import dirname, join

from seaqube.nlp.context2vec.context2vec import Context2Vec
from seaqube.tools.math import cosine

corpus = [['till', 'this', 'moment', 'i', 'never', 'knew', 'myself', '.'],
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

class TestContext2Vec(unittest.TestCase):
    def test_training(self):
        c2v = Context2Vec(epoch=1)
        c2v.train(corpus)
        c2v.save("c2v_0001")

    def test_loading(self):
        path = join(dirname(__file__), 'test_data', 'c2v_testmodel')
        c2v = Context2Vec.load(path)
        self.assertEqual(len(c2v.wv['of']), 600)
        first_a = c2v.backend_model.context2vec(['a', 'man', 'and', 'a', 'woman'], 0)
        second_a = c2v.backend_model.context2vec(['a', 'man', 'and', 'a', 'woman'], 3)
        self.assertAlmostEqual(cosine(first_a, second_a), 0.9895, delta=0.01)

