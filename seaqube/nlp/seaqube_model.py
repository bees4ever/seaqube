from os.path import basename

import dill
# some strange fix
import numpy
from nltk import word_tokenize

from seaqube.nlp._types import RawModelTinCan
from seaqube.tools.math import sif

dill._dill._reverse_typemap['ClassType'] = type



class SeaQuBeNLPToken:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector

    def __str__(self):
        return self.text

    def __repr__(self):
        return str(self)


class SeaQuBeNLPDoc:
    def __init__(self, docs, text, word_frequency):
        self.docs = docs
        self.original_text = text
        self.word_frequency = word_frequency

    def __str__(self):
        return self.original_text

    @property
    def text(self):
        return self.original_text

    @property
    def vector(self):
        return numpy.mean([doc.vector for doc in self.docs], axis=0)

    @property
    def sif_vector(self):
        return sif(self.word_frequency, [self.docs])

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        return self.docs[item]

    def __iter__(self):
        return iter(self.docs)

    def __len__(self):
        return len(self.docs)


class SeaQuBeNLP:
    def __init__(self, tin_can: RawModelTinCan, name):
        self.model = tin_can.model
        self.word_frequency = tin_can.word_frequency
        self.__wv = self.model.wv
        self.human_readable_name = name

    def w2v_embed(self, word):
        try:
            return self.__wv(word)
        except KeyError:
            return numpy.array(300 * [0.0])

    def vocab(self):
        return self.model.vocabs()

    def w2v(self, word):
        return SeaQuBeNLPToken(word, self.w2v_embed(word))

    def __call__(self, text):
        docs = [self.w2v(token.lower()) for token in word_tokenize(text) if token.isspace() is False]
        # print("debug", docs)
        return SeaQuBeNLPDoc(docs, text, self.word_frequency)

    def __str__(self):
        return f"CustomNLPDoc(model={self.model})@{hex(self.__hash__())}"

    def __repr__(self):
        return str(self)



class SeaQuBeNLPLoader:
    @staticmethod
    def load_model_from_path(path: str) -> SeaQuBeNLP:
        model = dill.load(open(path, "rb"))
        return SeaQuBeNLP(model, basename(path))

    @staticmethod
    def load_model_from_tin_can(tin_can: RawModelTinCan, name) -> SeaQuBeNLP:
        return SeaQuBeNLP(tin_can, name)


