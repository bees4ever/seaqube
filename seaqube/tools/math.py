"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns

NLP Math toolkit of SeaQuBe
"""

import itertools
import operator
import string
from functools import reduce  # Required in Python 3
from typing import Iterable

import numpy as np
from numpy import array, diff, sign, sort, math
import pandas
import numpy


def prod(iterable):
    """
    Stream-like multiply
    """
    return reduce(operator.mul, iterable, 1)


def critical_point(points):
    """
    Get 2nd derivation of distinct points
    Based on a dirty hack to use efficient sort method but with other order which is copied from
        https://stackoverflow.com/a/45877045/5885054.
    """
    if len(points) == 0:
        return 0.0
    points = -sort(-array(points))
    data = points[1:] - points[0:-1]
    a = diff(sign(diff(data))).nonzero()[0] + 1  # local min+max

    return points[a[0] - 1]


def cosine(a, b):
    """
    Cosine function using NumPy.
    """
    norm1 = numpy.linalg.norm(a)
    norm2 = numpy.linalg.norm(b)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0
    else:
        return a.dot(b) / (norm1 * norm2)


def sif(word_count, docs):
    """
    SIF - Similarity method based on Cosine with weighted values.
    Algorithm from: https://openreview.net/forum?id=SyK00v5xx.
    """
    if type(docs) not in [list, pandas.core.series.Series]:
        docs = [docs]
    all_words = sum(word_count.values())  # 54780
    a = 0.01

    dimension = 300
    if len(docs[0]) > 0:
        dimension = len(docs[0][0].vector)

    def p(w: str):
        return word_count[w] / all_words

    sentence_embedding = np.array([])

    for doc in docs:
        sentence = [d for d in doc if d.text not in string.punctuation]
        if len(sentence) > 0:
            sentence_vec = (1 / len(sentence)) * np.sum([(a / (a + p(w.text))) * w.vector for w in sentence],
                                                        axis=0).reshape(dimension)
            sentence_embedding = np.append(sentence_embedding, sentence_vec)
        else:
            sentence_embedding = np.append(sentence_embedding, np.array(dimension * [0.0]))

    lines = len(docs)

    sentence_embedding = np.transpose(sentence_embedding.reshape(lines, dimension))

    # First singular vector of singular vector decomposition of the matrix of all embeddings
    u = np.linalg.svd(sentence_embedding)[1]
    uu_t = np.dot(u, u)
    # do the matrix calculation
    for col in range(lines):
        v_s = sentence_embedding[:, col]
        sentence_embedding[:, col] = v_s - uu_t * v_s

    sif_embedding = np.transpose(sentence_embedding)
    if len(sif_embedding) == 1:
        sif_embedding = sif_embedding.reshape(prod(sif_embedding.shape))

    return sif_embedding


def f_score(precision, recall, score):
    """
    Classical F_\beta score calculation.
    """
    return (1 + score ** 2) * (precision * recall) / (score ** 2 * precision + recall)


def layz_pair_creation(a, b, random, max_len = 30):
    """
    Stream based and RAM efficient pair creation. Used for pairs of word's.
    However, it simply takes two list of elements.
    """
    ## the pair creator - always thing "n over k"
    pos = 0
    for k in range(1, len(b)):
        for elem in itertools.combinations(a, k):
            if pos >= max_len:
                break
            yield zip(elem, random.sample(b, k))
            pos += 1


def lazy_sample(population: Iterable, length: int, k: int, random):
    """
    Sample very big list is a lazy method to save RAM, based on Python's `random.sample` method.
    """
    if length > 1000000:
        length = 1000000

    if k > length:
        raise ValueError(f"k is greater then length, {k} > {length}, a sample is not possible. The max length is: 1000000")

    indecies = sorted(random.sample(range(length), k))

    try:
        next_index = indecies.pop(0)

        for i, sample in enumerate(population):
            if i == next_index:
                yield sample
                next_index = indecies.pop(0)
    except IndexError:
        pass
