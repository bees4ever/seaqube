"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

from seaqube.package_config import log
from seaqube.tools.types import Configable


class ChainTerm(Configable):
    """
    Wraps simple methods to a reduction augmenation.
    """
    def __init__(self, call: callable):
        self.call = call

    def get_config(self):
        return dict(method_name=str(self.call), note="wrapped by ChainTerm")

    def input_type(self):
        return "doc"

    def doc_augment(self, doc=None):
        return self.call(doc)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class CallOnOneChain(object):
    """
    Approach to chaining NLP augmentation one after another and NOT piping every doc to each augmentation.
    """
    def __init__(self, callables: list):
        self.callables: list = callables
        if len(self.callables) == 0:
            raise ValueError("Chain on empty call list is boring, please provide at least one callable")

    def __call__(self, parameter):
        first_caller = self.callables[0]
        log.info(f"Run first in chain: {str(first_caller)}")
        result = first_caller(parameter)
        log.debug(f"Debug result of first caller: {result}")

        for caller in self.callables[1:]:
            log.info(f"Run next in chain: {str(caller)}")
            result = caller(result)
            log.debug(f"Debug result of next and so on: {result}")

        return result
