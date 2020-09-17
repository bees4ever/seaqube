'''
Copyright (c) 2020 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
'''

from abc import abstractmethod

from seaqube.tools.types import Configable


class BaseReduction(Configable):
    def __str__(self):
        return str(self.__class__.__name__)

    def __repr__(self):
        return str(self)

    def input_type(self):
        """
        Defines the type which mode the augmentation method supports
        :return str: doc or text or corpus
        """
        return 'corpus'

    def __call__(self, corpus):
        return self.reduction_implementation(corpus)

    @abstractmethod
    def reduction_implementation(self, corpus):
        pass
