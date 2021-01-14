"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

from seaqube.augmentation.reduction._reduction import BaseReduction
from seaqube.nlp.tools import unique_2d_list


class UniqueCorpusReduction(BaseReduction):
    """
    A reduce class for the augmentation streaming mode. Reduce docs while remove duplicate docs.
    """
    def get_config(self):
        """
        Gives a dict with all relevant variables the object can recreated with (init parameters)
        Returns: dict of object config

        """
        return dict(method="make a corpus with unique doc, i.e. remove duplicate docs", class_name=str(self))

    def reduction_implementation(self, corpus):
        """
        Only keep those lists inside the big list which are unique.
        Args:
            corpus: list of list where second dimension can contain duplicates

        Returns: list of docs

        """
        return unique_2d_list(corpus)
