"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

from progressbar import progressbar

from seaqube.augmentation.misc.embedding_model_wrapper import PreTrainedModel
from seaqube.augmentation.misc.model_loader import load_fasttext_en_pretrained, load_word2vec_en_pretrained
from seaqube.benchmark._benchmark import BaseWordEmbeddingBenchmark, BenchmarkScore
from seaqube.nlp.tools import sentenceize_corpus
from seaqube.nlp.types import SeaQuBeWordEmbeddingsModel
from seaqube.package_config import log
from seaqube.tools.math import f_score

try:
    import vec4ir
except ImportError:
    raise ValueError("The vec4ir module is not available, please run: `from seaqube import download;download('vec4ir')`")


from vec4ir import WordCentroidDistance, Matching, Retrieval
from numpy import array, mean


class Corpus4IRBenchmark(BaseWordEmbeddingBenchmark):
    """
    An evaluation method which is based on the Word Centroid Method (vec4ir-package) to simulate an IR system and to
    evaluate the performance the word embeddings based on how the IR scores (F_1 score). The word embeddings are used to
    boost the IR System using the Word Centroid Method. Hence, the word embedding's  semantically correctness
    influences the performance of the IR system.

    But, the "good" and "correct" results of the IR system is based on an pre-trained model, because no real feedback
    exists. Therefore, failures of the pre-trained model influences the evaluation.
    """
    def __init__(self, small_corpus, model="w2v", threshold=0.9):
        if model not in ["ft", "w2v"]:
            raise ValueError("model can only be one of 'ft' or 'w2v'")

        if model == "ft":
            self.model: PreTrainedModel = load_fasttext_en_pretrained()
        elif model == "w2v":
            self.model: PreTrainedModel = load_word2vec_en_pretrained()

        self.threshold = threshold
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

    def __call__(self, model: SeaQuBeWordEmbeddingsModel) -> BenchmarkScore:
        custom_retrieval_model = self.setup_ir(model.wv, self.corpus)

        x = []
        y = {}
        for i in progressbar(range(len(self.corpus))):
            x.append((i, self.corpus[i]))

            result = self.retrieval.query(self.corpus[i], return_scores=True) # # original retrieval
            result_evaluate_x = list(zip(list(result[0]), list(result[1])))

            y[i] = {doc: score for (doc, score) in result_evaluate_x if score >= self.threshold}

        scores = custom_retrieval_model.evaluate(x, y)

        mean_scores = {k: mean(v) for k, v in scores.items()}

        return BenchmarkScore(mean_scores['f1_score'], dict(recall=mean_scores['recall'], precision=mean_scores['precision']))

    def get_config(self):
        return dict(class_name=str(self), model=self.model.__class__.__name__,)
