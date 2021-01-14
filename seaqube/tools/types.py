"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

from abc import ABC, abstractmethod


class Configable(ABC):
    @abstractmethod
    def get_config(self):
        """
        Gives a dict with all relevant variables the object can recreated with (init parameters)
        Returns: dict of object config

        """
        pass


class Writeable(ABC):
    """
     A simple interface which indicates what a `writeable` class needs which methods
    """
    @abstractmethod
    def write(self, data):
        """
        Write to file or buffer object
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close the file or end caching
        """
        pass
