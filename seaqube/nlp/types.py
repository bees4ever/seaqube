"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

import time
from abc import ABC, abstractmethod
from threading import Thread
from typing import List

import schedule
from seaqube.nlp.context2vec.context2vec import Context2Vec
from seaqube.tools.math import cosine


class SeaQuBeWordEmbeddingsModel(ABC):
    """
    Base SeaQuBeWordEmbeddings, which is needed for model evaluation and the interactive NLP wrapper.
    """
    def similarity(self, word_one, word_two) -> float:
        return cosine(self.wv[word_one], self.wv[word_two])

    @abstractmethod
    def vocabs(self) -> List[str]:
        pass

    @abstractmethod
    def word_vector(self, word):
        pass

    @property
    def wv(self):
        raise NotImplementedError()

    @abstractmethod
    def matrix(self):
        """
        Returns all word vectors as matrix
        Returns:
            Matrix of word vectors

        """
        pass


class RawModelTinCan(object):
    """
    Minimal NLP model transfer object for saving and loading SeaQuBe based NLP models.
    """
    def __init__(self, model: SeaQuBeWordEmbeddingsModel, word_frequency):
        self.model = model
        self.word_frequency = word_frequency


class BackgroundScheduler(Thread):
    """
    Toolkit for running background jobs.
    """
    def __init__(self):
        self.local_scheduler = schedule.Scheduler()
        super(BackgroundScheduler, self).__init__()
        self.nonstop = True

    def stop(self):
        self.nonstop = False


    def run(self) -> None:
        while self.nonstop:
            time.sleep(1)
            self.local_scheduler.run_pending()


class SeaQuBeWordEmbeddingsModelC2V(SeaQuBeWordEmbeddingsModel):
    """
    Context2Vec wrapper for SeaQuBe.
    """
    def __init__(self, c2v: Context2Vec):
        self.c2v = c2v

    def vocabs(self) -> List[str]:
        return self.c2v.wv.vocabs

    @property
    def wv(self):
        return self.c2v.wv

    def word_vector(self, word):
        return self.c2v.wv[word]

    def matrix(self):
        return self.c2v.wv.matrix


class SeaQuBeWordEmbeddingsModelGensim(SeaQuBeWordEmbeddingsModel):
    """
    Gensim (https://radimrehurek.com/gensim/) based NLP models wrapper for SeaQuBe.
    """
    def __init__(self, gensim_model):
        self.gensim_model = gensim_model

    def vocabs(self) -> List[str]:
        return list(self.gensim_model.wv.vocab.keys())

    @property
    def wv(self):
        return self.gensim_model.wv

    def word_vector(self, word):
        return self.gensim_model.wv[word]

    def matrix(self):
        return self.gensim_model.wv.vectors


class SeaQuBeWordEmbeddingsModelRawFT(SeaQuBeWordEmbeddingsModel):
    """
    Wrapper for SeaQuBe for official pre-trained fastText models.
    """
    def __init__(self, raw_ft):
        vocs = list(raw_ft[13])
        self.__wv = SeaQuBeNLPModel2WV(vocs, raw_ft[15][0: len(vocs)])

    def vocabs(self) -> List[str]:
        return self.__wv.vocabs

    @property
    def wv(self):
        return self.__wv

    def word_vector(self, word):
        return self.__wv[word]

    def matrix(self):
        return self.__wv.matrix


class SeaQuBeNLPModel2WV:
    """
    A interactive word embedding class, to easily get a word's embedding.
    """
    def __init__(self, vocabs: list, matrix):
        self.vocabs: list = vocabs
        self.matrix = matrix
        self.index2word = vocabs
        self.vectors = matrix

    def __getitem__(self, word):
        return self.matrix[self.vocabs.index(word)]


class SeaQuBeWordEmbeddingsModelCompressed(SeaQuBeWordEmbeddingsModel):
    """
    SeaQuBe's space efficient NLP's word embedding class.
    """
    def vocabs(self) -> List[str]:
        return self.wv_object.vocabs

    def word_vector(self, word):
        return self.wv_object[word]

    @property
    def wv(self):
        return self.wv_object

    def matrix(self):
        return self.wv_object.matrix

    def __init__(self, wv: SeaQuBeNLPModel2WV):
        self.wv_object: SeaQuBeNLPModel2WV = wv

