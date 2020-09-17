'''
Copyright (c) 2020 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
'''

from progressbar import progressbar

from seaqube.augmentation.misc.embedding_model_wrapper import PreTrainedModel
from seaqube.augmentation.misc.model_loader import load_fasttext_en_pretrained, load_word2vec_en_pretrained
from seaqube.benchmark._benchmark import BaseWordEmbeddingBenchmark, BenchmarkScore
from seaqube.nlp.tools import sentenceize_corpus
from seaqube.nlp.types import SeaQueBeWordEmbeddingsModel
from seaqube.package_config import log
from seaqube.tools.math import f_score
from vec4ir import WordCentroidDistance, Matching, Retrieval
from numpy import array


class Corpus4IRBenchmark(BaseWordEmbeddingBenchmark):
    def __init__(self, small_corpus, model="w2v"):
        if model not in ["ft", "w2v"]:
            raise ValueError("model can only be one of 'ft' or 'w2v'")

        if model == "ft":
            self.model: PreTrainedModel = load_fasttext_en_pretrained()
        elif model == "w2v":
            self.model: PreTrainedModel = load_word2vec_en_pretrained()

        self.corpus = sentenceize_corpus(small_corpus)
        self.retrieval = self.setup_ir(self.model.wv, self.corpus)

    def method_name(self):
        return "corpus4ir"

    @staticmethod
    def setup_ir(wv, corpus):
        wcd = WordCentroidDistance(wv)
        wcd.fit(corpus)
        match_op = Matching().fit(corpus)
        retrieval = Retrieval(wcd, matching=match_op)
        retrieval.fit(corpus)
        return retrieval

    @staticmethod
    def perform_ir(retrieval, query):
        result = array(retrieval.query(query, return_scores=True))
        return result[0][result[1] > 0.75]  # 0.75 is a constant

    def __call__(self, model: SeaQueBeWordEmbeddingsModel) -> BenchmarkScore:
        custom_model_retrival = self.setup_ir(model.wv, self.corpus)
        tp, fn, fp = 0, 0, 0

        for sentence in progressbar(self.corpus):

            relevant_documents = set(self.perform_ir(self.retrieval, sentence))
            search_result_documents = set(self.perform_ir(custom_model_retrival, sentence))

            tp += len(relevant_documents.intersection(search_result_documents))
            fn += len(relevant_documents) - len(relevant_documents.intersection(search_result_documents))
            fp += len(search_result_documents.difference(relevant_documents))
            log.debug(f"{self.__class__.__name__}: tp, fn, fp:{tp}, {fn}, {fp}")
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

        return BenchmarkScore(f_score(precision, recall, 1), dict(tp=tp, fn=fn, fp=fp))

    def get_config(self):
        return dict(class_name=str(self), model=self.model.__class__.__name__,)

