"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

from abc import ABC, abstractmethod
from typing import List

from seaqube.nlp.types import SeaQuBeNLPModel2WV, SeaQuBeWordEmbeddingsModel


class PreTrainedModel(SeaQuBeWordEmbeddingsModel):
    @abstractmethod
    def similar_by_word(self, word, topn=10):
        pass


class PreTrainedFTRawEN(SeaQuBeWordEmbeddingsModel):
    """
    Wraps the gensim fastText pretrained model to a SeaQuBeWordEmbeddings
    """
    def __init__(self, raw_model):
        self.raw_model = raw_model
        vocabs = list(self.raw_model[13].keys())
        self.__wv = SeaQuBeNLPModel2WV(vocabs, self.raw_model[15][0:len(vocabs)])

    @property
    def wv(self):
        return self.__wv

    def vocabs(self) -> List[str]:
        return self.__wv.vocabs

    def word_vector(self, word):
        return self.__wv[word]

    def matrix(self):
        return self.__wv.matrix


class PreTrainedGensimEN(SeaQuBeWordEmbeddingsModel):
    """
    Wraps the gensim word2vec pretrained model to a SeaQuBeWordEmbeddings
    """
    def __init__(self, loaded_model):
        self.loaded_model = loaded_model

    def similar_by_word(self, word, topn=10):
        return self.loaded_model.similar_by_word(word, topn)

    @property
    def wv(self):
        return self.loaded_model.wv

    def vocabs(self) -> List[str]:
        return list(self.loaded_model.wv.vocab)

    def word_vector(self, word):
        return self.loaded_model.wv[word]

    def matrix(self):
        return self.loaded_model.wv.vectors


