'''
Copyright (c) 2020 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
'''

from pandas import read_csv
from progressbar import progressbar
from scipy.stats import pearsonr, shapiro, spearmanr, kendalltau
from seaqube.benchmark._benchmark import DataSetBasedWordEmbeddingBenchmark, get_shipped_test_set_path, \
    get_list_of_shipped_test_sets, BenchmarkScore
from seaqube.nlp.types import SeaQuBeWordEmbeddingsModel
from seaqube.package_config import log


class WordSimilarityBenchmark(DataSetBasedWordEmbeddingBenchmark):
    def method_name(self):
        return "wordsimilarity"

    def _load_shipped_test_set(self, test_set):
        """"

        """
        path = get_shipped_test_set_path("word-similarity", test_set)
        return read_csv(path)

    def available_test_sets(self):
        return get_list_of_shipped_test_sets("word-similarity")

    def __call__(self, model: SeaQuBeWordEmbeddingsModel) -> BenchmarkScore:
        model_sim, sheet_sim = [], []
        for rowitem in progressbar(self.test_set.iterrows(), max_value=len(self.test_set)):
            _, row = rowitem

            if row.word1 in model.vocabs() and row.word2 in model.vocabs():
                model_sim.append(model.similarity(row.word1, row.word2))
                sheet_sim.append(row.similarity)
        log.info(
            f"{self.__class__.__name__}: Correlation of list of length={len(model_sim)}")
        if len(model_sim) < 2:
            return BenchmarkScore(0.0)

        p_corr = pearsonr(model_sim, sheet_sim)
        s_corr = spearmanr(model_sim, sheet_sim)
        k_corr = kendalltau(model_sim, sheet_sim)

        payload = {'pearson':
                   {
                       'correlation': p_corr[0], 'pvalue': p_corr[1]
                   },
                   'spearman':
                   {
                       'correlation': s_corr.correlation, 'pvalue': s_corr.pvalue
                   },
                       'kendalltau': {
                           'correlation': k_corr.correlation, 'pvalue': k_corr.pvalue
                   },
                   'shapiro': {
                       'original_sim': {
                           'pvalue': shapiro(sheet_sim)[0], 'statistic': shapiro(sheet_sim)[1]
                       },
                       'model_sim': {
                           'pvalue': shapiro(model_sim)[0], 'statistic': shapiro(model_sim)[1]
                       }
                   },
                   'matched_words': len(model_sim)}

        return BenchmarkScore(p_corr[0], payload)
