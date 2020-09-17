from pandas import read_csv
from progressbar import progressbar
from scipy.stats import pearsonr

from seaqube.benchmark._benchmark import DataSetBasedWordEmbeddingBenchmark, get_shipped_test_set_path, \
    get_list_of_shipped_test_sets
from seaqube.nlp._types import SeaQueBeWordEmbeddingsModel


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

    def __call__(self, model: SeaQueBeWordEmbeddingsModel):
        model_sim, sheet_sim = [], []
        for rowitem in progressbar(self.test_set.iterrows()):
            _, row = rowitem

            if row.word1 in model.vocabs() and row.word2 in model.vocabs():
                model_sim.append(model.similarity(row.word1, row.word2))
                sheet_sim.append(row.similarity)
        log.info(f"{self.__class__.__name__}: Correlation of list of length={len(model_sim)}")
        if len(model_sim) < 2:
            return 0.0

        return pearsonr(model_sim, sheet_sim)

