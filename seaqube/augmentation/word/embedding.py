"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

import copy

import random
from itertools import product
from typing import Iterable

from seaqube.augmentation.base import MultiprocessingAugmentation
from seaqube.augmentation.misc.embedding_model_wrapper import PreTrainedModel
from seaqube.augmentation.misc.model_loader import load_fasttext_en_pretrained, load_word2vec_en_pretrained
from seaqube.package_config import log
from seaqube.tools.math import lazy_sample


class EmbeddingAugmentation(MultiprocessingAugmentation):
    """
        "TinyBERT: Distilling BERT for Natural Language Understanding" of Jiao et al., (https://arxiv.org/pdf/1909.10351.pdf) presents the idea to
        augment text corpora by replacing similar words based on word embeddings. Therefor pretrained models
        are needed.


        @misc{jiao2019tinybert,
            title={TinyBERT: Distilling BERT for Natural Language Understanding},
            author={Xiaoqi Jiao and Yichun Yin and Lifeng Shang and Xin Jiang and Xiao Chen and Linlin Li and Fang Wang and Qun Liu},
            year={2019},
            eprint={1909.10351},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        }

    """

    def __init__(self, model="w2v", similar_n=3, max_length: int = 100, remove_duplicates: bool = False,
                 multiprocess: bool = True, seed: int = None):
        """

        Args:
            model: which pretrained model should be used for similar replacement
            similar_n:
            max_length: cut the produced text at a limit to prevent overflow
            remove_duplicates: remove after augmentation for duplicates
            multiprocess: if augmentation class implements the multiprocessing call, then it can be turn off again with
                    this flag, most for testing purpose
            seed: fix the randomness with a seed for testing purpose
        """
        if model not in ["ft", "w2v"]:
            raise ValueError("model can only be one of 'ft' or 'w2v'")

        log.info(f"{self.__class__.__name__}: Start load model")
        if model == "ft":
            self.model: PreTrainedModel = load_fasttext_en_pretrained()
        elif model == "w2v":
            self.model: PreTrainedModel = load_word2vec_en_pretrained()

        log.info(f"{self.__class__.__name__}: End load model")
        self.max_length = max_length
        self.remove_duplicates = remove_duplicates
        self.multiprocess = multiprocess
        self.seed = seed
        self.r = random.Random()

        if seed is not None:
            self.r.seed(seed)

        self.similar_n = similar_n

    def get_config(self):
        """
        Gives a dict with all relevant variables the object can recreated with (init parameters)
        Returns: dict of object config

        """
        return dict(model=str(self.model), similar_n=self.similar_n, max_length=self.max_length,
                    remove_duplicates=self.remove_duplicates, seed=self.seed, class_name=str(self))

    def shortname(self):
        return "embedding"

    def input_type(self):
        """
        Which return type is supported
        Returns: doc or text
        """
        return "doc"

    def augmentation_implementation(self, doc: list):

        augmented = []
        similars = []
        for word in doc:
            try:
                similars.append([None] + list(map(lambda x: x[0], self.model.similar_by_word(word)))[0:self.similar_n])
            except KeyError:
                similars.append([None])

        combinations = product(range(self.similar_n + 1), repeat=len(doc))
        combinations_sample = lazy_sample(combinations, (self.similar_n + 1)**len(doc), self.max_length, self.r)

        for tuple_ in combinations_sample:
            tmp_doc = copy.deepcopy(doc)
            for word_pos, index in enumerate(tuple_):
                try:
                    if similars[word_pos][index] is not None:
                        tmp_doc[word_pos] = similars[word_pos][index]
                except IndexError:
                    pass

            augmented.append(tmp_doc)

        return augmented
