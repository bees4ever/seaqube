"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

import gc
from numpy import log as np_log, array
import operator
import random

from seaqube.augmentation.base import MultiprocessingAugmentation
from seaqube.nlp.tools import word_count_list
from seaqube.tools.math import layz_pair_creation


class TfIdf:
    """
    Simple TfIdf calculation based on a given corpus. Former invented by Karen Spärck Jones FBA (26 August 1935 – 4 April 2007)
    """
    def __init__(self, corpus):
        self.corpus = corpus

    def tf(self, word, doc):
        count = word_count_list([doc])
        return count[word] / len(doc)

    def idf(self, word):
        N = len(self.corpus)
        df = len(list(filter(lambda doc: word in doc, self.corpus)))
        if df == 0:
            return 0
        return np_log(N / df)

    def tf_idf(self, doc):
        df = dict()
        for word in doc:
            df[word] = self.tf(word, doc) * self.idf(word)
        return df


class UnigramAugmentation(MultiprocessingAugmentation):
    def __init__(self, corpus,  n=3, find_threshold=0.7, replace_threshold=0.8, max_length: int = 100,
                 remove_duplicates: bool = False, multiprocess=True, seed: int = None):
        """

           Args:
               corpus: original corpus where augmentation is performed on
               n: max number of how many words are replaced
               find_threshold: probability when a word is selected
               replace_threshold: probability when a word is replaced. Default is 0.8, if value is None,
                then it will be sampled
               max_length: cut the produced text at a limit to prevent overflow
               remove_duplicates: remove after augmentation for duplicates
                multiprocess: if augmentation class implements the multiprocessing call, then it can be turn off again with
                    this flag, most for testing purpose
               seed: fix the randomness with a seed for testing purpose

        """

        self.tfidf = TfIdf(corpus)
        self.count = word_count_list(corpus)
        self.words = list(self.count.keys())
        self.multiprocess = multiprocess
        self.max_length = max_length
        self.remove_duplicates = remove_duplicates
        self.seed = seed
        self.random = random.Random()

        if self.seed is not None:
            self.random.seed(self.seed)

        self.n = n
        self.find_threshold = find_threshold
        self.replace_threshold = replace_threshold

    def __del__(self):
        """
        Destructor should tidy up all corpus based activities
        """
        del self.tfidf
        del self.count
        del self.words
        gc.collect()

    def get_config(self):
        """
        Gives a dict with all relevant variables the object can recreated with (init parameters)
        Returns: dict of object config

        """
        return dict(n=self.n, find_threshold=self.find_threshold, replace_threshold=self.replace_threshold,
                    max_length=self.max_length, remove_duplicates=self.remove_duplicates, seed=self.seed,
                    class_name=str(self))

    def shortname(self):
        return "unigram"

    def input_type(self):
        return "doc"

    def augmentation_implementation(self, doc):
        """
        Algorithm unigram is published in "Unsupervised Data Augmentation for Consistency Training" by Qizhe Xie and Zihang Dai and Eduard Hovy and Minh-Thang Luong and Quoc V. Le,
        where Unigram is one among other algorithms.


        @misc{xie2019unsupervised,
            title={Unsupervised Data Augmentation for Consistency Training},
            author={Qizhe Xie and Zihang Dai and Eduard Hovy and Minh-Thang Luong and Quoc V. Le},
            year={2019},
            eprint={1904.12848},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
        }

        Args:
            doc: a tokenized sentences, called doc, i.e.: [I have a dream]

        Returns: list of augmented docs, based on input

        """

        df_2 = self.tfidf.tf_idf(doc)

        max_tfidf_doc1 = max(df_2.items(), key=operator.itemgetter(1))[0]
        C = df_2[max_tfidf_doc1]

        tfidfs = array(list(df_2.values()))
        Z = sum(C - tfidfs) / len(doc)

        #P = min(p * (C - tfidfs) / Z)

        to_replace_indecies = array(range(len(df_2)))[self.find_threshold * (C - tfidfs) / Z >= 1]

        scored = {word: self.s(word) for word in self.words}
        # the interessting thing is
        # _Z_ = sum(scored.values())
        _Z_ = max(scored.values())
        scored = {word: s / _Z_ for word, s in scored.items()}

        scored = {k: v for k, v in sorted(scored.items(), key=lambda item: item[1], reverse=True)}
        # No I selected a population
        if self.replace_threshold is None:
            self.replace_threshold = self.random.random()

        words_population = [word for word, score in scored.items() if score > self.replace_threshold]

        if len(words_population) < self.n:
            self.n = len(words_population)


        population_replace = self.random.sample(words_population, self.n)  # amount of how many words replaced

        docs = [doc]

        for pairs in layz_pair_creation(to_replace_indecies, population_replace, self.random, max_len=self.max_length):
            doc_augment = array(doc)
            for index, word in pairs:
                doc_augment[index] = word
            docs.append(list(doc_augment))

        return docs[0: self.max_length]

    def s(self, word):
        """
        Calculates the score S based on the tf-idf score
        """
        freq = self.count[word]
        return freq * self.tfidf.idf(word)
