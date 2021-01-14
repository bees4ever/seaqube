"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

from typing import Iterable

from progressbar import ProgressBar
from seaqube.benchmark._benchmark import BaseWordEmbeddingBenchmark, BenchmarkScore, complete_correlation_calculation
from seaqube.nlp.types import SeaQuBeWordEmbeddingsModel
from itertools import product, combinations, chain
from progressbar import progressbar
from nltk.corpus import wordnet
from scipy.special import comb

from seaqube.tools.umproc import ForEach


class SemanticWordnetBenchmark(BaseWordEmbeddingBenchmark):
    """
    The Semantic Networks Evaluation uses the WordNet (https://wordnet.princeton.edu/) Python
    library called NLTK2. The ‘Wu-Palmer-Similarity’ is used to calculate similarities between synsets inside a WordNet
    graph which is used as the ground of truth for the similarity of two words. This similarity is used to compare
    the similarity score of two words using the word embeddings which the object of evaluation.
    """
    def __init__(self, word_pairs: Iterable, multiprocessing: bool = False, max_cpus=None):
        self.word_pairs = word_pairs
        self.multiprocessing = multiprocessing
        self.max_cpus = max_cpus
        super(SemanticWordnetBenchmark, self).__init__()

    def get_config(self):
        return dict(class_name=str(self), word_pairsn=str(self.word_pairs))

    @staticmethod
    def word_pairs_from_vocab_list(vocabs: list):
        v_len = len(vocabs)
        return chain(combinations(vocabs, 2), zip(vocabs, vocabs)), int(comb(v_len, 2) + v_len)


    def method_name(self):
        """
        Return a human readable name of the benchmark method
        Returns str:
        """
        "semantic_wordnet"

    @staticmethod
    def wordnet_similarity(word1, word2):
        """
        Time consuming walking through the graph
        """
        syns1 = wordnet.synsets(word1)
        syns2 = wordnet.synsets(word2)
        sims = []

        if len(syns1) == 0 or len(syns2) == 0:
            return 0.0

        for sense1, sense2 in product(syns1, syns2):
            distance = wordnet.wup_similarity(sense1, sense2)
            distance_val = distance if distance is not None else 0.0

            # sims.append((d, syns1, syns2))
            sims.append(distance_val)

        return max(sims)

    def multi_proc_wordnet_sim(self, wordpairs):
        a, b = wordpairs
        return self.wordnet_similarity(a, b)


    def __call__(self, model: SeaQuBeWordEmbeddingsModel) -> BenchmarkScore:
        model_sim, graph_sim = [], []

        # wordnet_similarity is time consuming, seperate it in two steps

        filtered_pairs = []
        for word_one, word_two in progressbar(self.word_pairs):
            if word_one in model.vocabs() and word_two in model.vocabs():
                filtered_pairs.append((word_one, word_two))
                model_sim.append(model.similarity(word_one, word_two))

        if self.multiprocessing:
            prg = ProgressBar(max_value=len(filtered_pairs))
            multi_wrapper = ForEach(self.multi_proc_wordnet_sim, max_cpus=self.max_cpus)
            for score in multi_wrapper(filtered_pairs):
                graph_sim.append(score)
                prg.update(prg.value + 1)

        else:
            for word_one, word_two in progressbar(filtered_pairs):
                graph_sim.append(SemanticWordnetBenchmark.wordnet_similarity(word_one, word_two))

        if len(model_sim) < 2:
            return BenchmarkScore(0.0)

        return complete_correlation_calculation(model_sim, graph_sim)
