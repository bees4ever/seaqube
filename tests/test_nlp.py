import unittest
from os.path import dirname, join

from seaqube.nlp.seaqube_model import SeaQuBeCompressLoader


def model_path():
    return join(dirname(__file__), 'test_data', 'small_copressed_model.dill')

class TestNLP(unittest.TestCase):
    def test_api(self):
        nlp = SeaQuBeCompressLoader.load_compressed_model(model_path(), 'test_model')
        doc = nlp("I don’t know where you get your delusions, laserbrain")
        self.assertEqual(len(doc), 12)

        self.assertAlmostEquals(
            doc.similarity("The Emperor is not as forgiving as I am."),
            0,
            delta=0.1
        )

        self.assertAlmostEquals(
            doc.similarity("The Emperor is not as forgiving as I am.", vector="sif"),
            0,
            delta=0.1
        )

        # similarity for words
        word = doc[1]
        self.assertAlmostEquals(
            word.similarity("Vader"),
            0,
            delta=0.1
        )



