from abc import ABC, abstractmethod


class PreTrainedModel(ABC):
    @abstractmethod
    def similar_by_word(self, word, topn=10):
        pass

    @property
    @abstractmethod
    def wv(self):
        pass


class PreTrainedGensimEN(PreTrainedModel):
    def __init__(self, loaded_model):
        self.loaded_model = loaded_model

    def similar_by_word(self, word, topn=10):
        return self.loaded_model.similar_by_word(word, topn)

    @property
    def wv(self):
        return self.loaded_model.wv


