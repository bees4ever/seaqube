from pandas import read_csv
import numpy as np
from progressbar import progressbar

from seaqube.benchmark._benchmark import DataSetBasedWordEmbeddingBenchmark, get_shipped_test_set_path, \
    get_list_of_shipped_test_sets
from seaqube.nlp._types import SeaQueBeWordEmbeddingsModel


class WordAnalogyBenchmark(DataSetBasedWordEmbeddingBenchmark):
    def method_name(self):
        return "wordanalogy"

    def _load_shipped_test_set(self, test_set):
        """"

        """
        path = get_shipped_test_set_path("word-analogy", test_set)
        return read_csv(path)

    def available_test_sets(self):
        return get_list_of_shipped_test_sets("word-analogy")

    def most_similar(self, calculated_vector, model: SeaQueBeWordEmbeddingsModel, topn=10):
        vocab_len = len(model.vocabs())

        distances = np.array([np.linalg.norm(calculated_vector - model.matrix()[i]) for i in range(vocab_len)])
        found_indecies = np.array(distances).argsort()[0:topn]

        return list(zip(np.array(model.vocabs())[found_indecies], distances[found_indecies]))

    def __call__(self, model: SeaQueBeWordEmbeddingsModel):
        considered_lines = 0
        correct_hits = 0

        for rowitem in progressbar(self.test_set.iterrows()):
            _, row = rowitem
            if row.word1 in model.vocabs() and row.word2 in model.vocabs() and row.word3 in model.vocabs() and row.target in model.vocabs():
                considered_lines += 1  # all words need to be in the vocab list, otherwise it makes no sense
                calculated_wv = model.wv(row.word1) - model.wv(row.word2) + model.wv(row.word3)
                word = self.most_similar(calculated_wv, model)[0][0]
                correct_hits += int(word == row.target)

        if considered_lines == 0:
            return 0
            
        return correct_hits / considered_lines
