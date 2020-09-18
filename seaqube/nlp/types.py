'''
Copyright (c) 2020 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
'''

import time
from abc import ABC, abstractmethod
from threading import Thread
from typing import List

import schedule
from seaqube.tools.math import cosine


class SeaQueBeWordEmbeddingsModel(ABC):
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
    def __init__(self, model: SeaQueBeWordEmbeddingsModel, word_frequency):
        self.model = model
        self.word_frequency = word_frequency


class BackgroundScheduler(Thread):
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




class SeaQueBeWordEmbeddingsModelGensim(SeaQueBeWordEmbeddingsModel):
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


class SeaQuBeNLPModel2WV:
    def __init__(self, vocabs: list, matrix):
        self.vocabs: list = vocabs
        self.matrix = matrix
        self.index2word = vocabs
        self.vectors = matrix

    def __getitem__(self, word):
        return self.matrix[self.vocabs.index(word)]


class SeaQueBeWordEmbeddingsModelCompressed(SeaQueBeWordEmbeddingsModel):
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




class BenchmarkResult:
    def __init__(self, pre_recall, reports):
        self.pre_recall = pre_recall
        self.reports = reports


