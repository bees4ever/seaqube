'''
Copyright (c) 2020 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
'''

from abc import ABC, abstractmethod

from seaqube.nlp.types import SeaQuBeNLPModel2WV


class PreTrainedModel(ABC):
    @abstractmethod
    def similar_by_word(self, word, topn=10):
        pass

    @property
    @abstractmethod
    def wv(self):
        pass

class PreTrainedFTRawEN():
    def __init__(self, raw_model):
        self.raw_model = raw_model

    #self.loaded_model.similar_by_word(word, topn)

    @property
    def wv(self):
        return SeaQuBeNLPModel2WV(list(self.raw_model[13].keys()), self.raw_model[15])

class PreTrainedGensimEN(PreTrainedModel):
    def __init__(self, loaded_model):
        self.loaded_model = loaded_model

    def similar_by_word(self, word, topn=10):
        return self.loaded_model.similar_by_word(word, topn)

    @property
    def wv(self):
        return self.loaded_model.wv


