"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

from abc import abstractmethod
from os.path import basename

import dill
# some strange fix
from seaqube.nlp.types import SeaQuBeNLPModel2WV, RawModelTinCan, SeaQuBeWordEmbeddingsModelCompressed
from seaqube.nlp.tools import word_count_list

dill._dill._reverse_typemap['ClassType'] = type

import numpy
from nltk import word_tokenize
import multiprocessing
from seaqube.tools.math import sif, cosine
from seaqube.tools.types import Configable


class SeaQuBeNLPToken:
    def __init__(self, text, vector, nlp):
        self.text = text
        self.vector = vector
        self.nlp = nlp

    def similarity(self, word):
        if type(word) == str:
            doc = self.nlp(word)
            return cosine(self.vector, doc.vector)
        else:
            raise NotImplemented("Other input then text is not supported, yet")

    def __str__(self):
        return self.text

    def __repr__(self):
        return str(self)


class SeaQuBeNLPDoc:
    def __init__(self, docs, text, word_frequency, nlp):
        # nlp is of type SeaQuBeNLP
        self.nlp = nlp
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

    def similarity(self, text, vector="mean"):
        doc = None
        if type(text) == str:
            doc = self.nlp(text)
        else:
            raise NotImplemented("Other input then text is not supported, yet")

        if vector == "mean":
            return cosine(self.vector, doc.vector)
        elif vector == "sif":
            return cosine(self.sif_vector, doc.sif_vector)
        else:
            raise NotImplemented("One vector types [mean, sif] are implemented")

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
            return self.__wv[word]
        except KeyError:
            return numpy.array(self._dimension() * [0.0])
        except ValueError:
            return numpy.array(self._dimension() * [0.0])

    def _dimension(self):
        return self.model.matrix().shape[1]

    def vocab(self):
        return self.model.vocabs()

    def w2v(self, word):
        return SeaQuBeNLPToken(word, self.w2v_embed(word), self)

    def __call__(self, text):
        docs = [self.w2v(token.lower()) for token in word_tokenize(text) if token.isspace() is False]
        return SeaQuBeNLPDoc(docs, text, self.word_frequency, self)

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


class SeaQuBeCompressLoader:
    @staticmethod
    def save_model_compressed(tin_can: RawModelTinCan, path) -> None:
        #cc['matrix'][cc['vocab'].index("man")]
        compressed_model = {'vocabs': tin_can.model.vocabs(), 'matrix': tin_can.model.matrix(),
                            'wf': tin_can.word_frequency}
        with open(path, "wb") as f:
            dill.dump(compressed_model, f)
    
    @staticmethod
    def load_compressed_model(path: str, name):
        with open(path, "rb") as f:
            compressed_model = dill.load(f)

        model = SeaQuBeWordEmbeddingsModelCompressed(SeaQuBeNLPModel2WV(compressed_model['vocabs'],
                                                                         compressed_model['matrix']))

        tin_can = RawModelTinCan(model, compressed_model['wf'])

        return SeaQuBeNLP(tin_can, name)


class BaseModelWrapper(Configable):
    def __init__(self, max_cpus=None):
        self.epochs = -1
        self.model = None
        self.__processed = False
        self.data = None
        self.max_cpus = max_cpus

    @abstractmethod
    def define_model(self):
        pass

    @property
    def cpus(self):
        cpus = multiprocessing.cpu_count()
        
        # limit the number of cpu, if inside a shared machine or something else 
        if self.max_cpus is None:
            return cpus

        if cpus > self.max_cpus:
            return 64
        return cpus

    @property
    def name(self):
        return str(self.__class__.__name__)

    @abstractmethod
    def define_training(self):
        pass

    @abstractmethod
    def define_epochs(self):
        pass

    def train_on_corpus(self, data):
        self.epochs = self.define_epochs()
        self.data = data
        self.model = self.define_model()
        self.define_training()
        self.__processed = True

    @abstractmethod
    def _wrap_nlp_model(self, model):
        pass

    def get(self):
        if not self.__processed:
            raise ValueError("First run `process` otherwise the model is empty")

        return RawModelTinCan(self._wrap_nlp_model(self.model), word_count_list(self.data))


