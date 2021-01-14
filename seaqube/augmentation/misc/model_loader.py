"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

import os.path as Path
from gensim.models._fasttext_bin import load
import gensim.downloader as api

from seaqube.augmentation.misc.embedding_model_wrapper import PreTrainedGensimEN, PreTrainedFTRawEN
from seaqube.package_config import package_path, log


def load_fasttext_en_pretrained():
    """
    Load the gensim fasttext pretrained model. Is used to warp the model later on
    """
    log.info("Load FT Model")
    path = Path.join(package_path, 'augmentation', 'data', 'fasttext_en', 'cc.en.300.bin')

    if not Path.isfile(path):
        raise ValueError("Fast Text Pretrained Model is not available, please run: `from seaqube import download;download('fasttext-en-pretrained')`")

    with open(path, 'rb') as fin:
        return PreTrainedFTRawEN(load(fin))


def load_word2vec_en_pretrained():
    """
    Load the gensim word2vec pretrained model. Is used to warp the model later on
    """
    log.info("Load W2V Model")
    model = api.load("glove-wiki-gigaword-50")
    return PreTrainedGensimEN(model)
