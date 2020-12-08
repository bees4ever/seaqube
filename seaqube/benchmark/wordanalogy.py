'''
Copyright (c) 2020 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
'''
import gc
import itertools
from collections import Counter, OrderedDict

import six
from pandas import read_csv
import numpy as np
from progressbar import progressbar, ProgressBar
from six import string_types, iteritems
from sklearn.metrics import pairwise_distances

from seaqube.benchmark._benchmark import DataSetBasedWordEmbeddingBenchmark, get_shipped_test_set_path, \
    get_list_of_shipped_test_sets, BenchmarkScore
from seaqube.nlp.types import SeaQuBeWordEmbeddingsModel
from seaqube.package_config import log
from seaqube.tools.math import cosine
from seaqube.tools.umproc import ForEach

from six import text_type as unicode
'''
CountedVocabulary, OrderedVocabulary and Embeddingare part of ''
'''

class Vocabulary(object):
    """ A set of words/tokens that have consistent IDs.

    Attributes:
      word_id (dictionary): Mapping from words to IDs.
      id_word (dictionary): A reverse map of `word_id`.
    """

    def __init__(self, words=None):
        """ Build attributes word_id and id_word from input.

        Args:
          words (list/set): list or set of words.
        """
        words = self.sanitize_words(words)
        self.word_id = {w: i for i, w in enumerate(words)}
        self.id_word = {i: w for w, i in iteritems(self.word_id)}

    def __iter__(self):
        """Iterate over the words in a vocabulary."""
        for w, i in sorted(iteritems(self.word_id), key=lambda wc: wc[1]):
            yield w

    @property
    def words(self):
        """ Ordered list of words according to their IDs."""
        return list(self)

    def __unicode__(self):
        return u"\n".join(self.words)

    def __str__(self):
        if six.PY3:
            return self.__unicode__()
        return self.__unicode__().encode("utf-8")

    def __getitem__(self, key):
        if isinstance(key, string_types) and not isinstance(key, unicode):
            key = unicode(key, encoding="utf-8")
        return self.word_id[key]

    def add(self, word):
        if isinstance(word, string_types) and not isinstance(word, unicode):
            word = unicode(word, encoding="utf-8")

        if word in self.word_id:
            raise RuntimeError("Already existing word")

        id = len(self.word_id)
        self.word_id[word] = id
        self.id_word[id] = word

    def __contains__(self, key):
        return key in self.word_id

    def __delitem__(self, key):
        """Delete a word from vocabulary.

        Note:
         To maintain consecutive IDs, this operation implemented
         with a complexity of \\theta(n).
        """
        del self.word_id[key]
        self.id_word = dict(enumerate(self.words))
        self.word_id = {w: i for i, w in iteritems(self.id_word)}

    def __len__(self):
        return len(self.word_id)

    def sanitize_words(self, words):
        """Guarantees that all textual symbols are unicode.
        Note:
          We do not convert numbers, only strings to unicode.
          We assume that the strings are encoded in utf-8.
        """
        _words = []
        for w in words:
            if isinstance(w, string_types) and not isinstance(w, unicode):
                _words.append(unicode(w, encoding="utf-8"))
            else:
                _words.append(w)
        return _words

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError as e:
            return default

    def getstate(self):
        return list(self.words)

class OrderedVocabulary(Vocabulary):
    """ An ordered list of words/tokens according to their frequency.

    Note:
      The words order is assumed to be sorted according to the word frequency.
      Most frequent words appear first in the list.

    Attributes:
      word_id (dictionary): Mapping from words to IDs.
      id_word (dictionary): A reverse map of `word_id`.
    """

    def __init__(self, words=None):
        """ Build attributes word_id and id_word from input.

        Args:
          words (list): list of sorted words according to frequency.
        """

        words = self.sanitize_words(words)
        self.word_id = {w: i for i, w in enumerate(words)}
        self.id_word = {i: w for w, i in iteritems(self.word_id)}

    def most_frequent(self, k):
        """ Returns a vocabulary with the most frequent `k` words.

        Args:
          k (integer): specifies the top k most frequent words to be returned.
        """
        return OrderedVocabulary(words=self.words[:k])

class CountedVocabulary(OrderedVocabulary):
    """ List of words and counts sorted according to word count.
    """

    def __init__(self, word_count=None):
        """ Build attributes word_id and id_word from input.

        Args:
          word_count (dictionary): A dictionary of the type word:count or
                                   list of tuples of the type (word, count).
        """

        if isinstance(word_count, dict):
            word_count = iteritems(word_count)
        sorted_counts = list(sorted(word_count, key=lambda wc: wc[1], reverse=True))
        words = [w for w, c in sorted_counts]
        super(CountedVocabulary, self).__init__(words=words)
        self.word_count = OrderedDict(sorted_counts)

    def most_frequent(self, k):
        """ Returns a vocabulary with the most frequent `k` words.

        Args:
          k (integer): specifies the top k most frequent words to be returned.
        """
        word_count = [(w, self.word_count[w]) for w in self.words[:k]]
        return CountedVocabulary(word_count=word_count)

    def min_count(self, n=1):
        """ Returns a vocabulary after eliminating the words that appear < `n`.

        Args:
          n (integer): specifies the minimum word frequency allowed.
        """
        word_count = [(w, c) for w, c in iteritems(self.word_count) if c >= n]
        return CountedVocabulary(word_count=word_count)

    def __unicode__(self):
        return u"\n".join([u"{}\t{}".format(w, self.word_count[w]) for w in self.words])

    def __delitem__(self, key):
        super(CountedVocabulary, self).__delitem__(key)
        self.word_count = OrderedDict([(w, self.word_count[w]) for w in self])

    def getstate(self):
        words = list(self.words)
        counts = [self.word_count[w] for w in words]
        return (words, counts)

    @staticmethod
    def from_vocabs(vocabs):
        word_count = Counter()

        for token in vocabs:
            word_count[token] += 1

        return CountedVocabulary(word_count=word_count)


class Embedding(object):
    """ Mapping a vocabulary to a d-dimensional points."""

    def __init__(self, vocabulary, vectors):
        self.vocabulary = vocabulary
        self.vectors = np.asarray(vectors)
        if len(self.vocabulary) != self.vectors.shape[0]:
            raise ValueError("Vocabulary has {} items but we have {} "
                             "vectors."
                             .format(len(vocabulary), self.vectors.shape[0]))

        if len(self.vocabulary.words) != len(set(self.vocabulary.words)):
            logger.warning("Vocabulary has duplicates.")

    def __getitem__(self, k):
        return self.vectors[self.vocabulary[k]]

    def __setitem__(self, k, v):
        if not v.shape[0] == self.vectors.shape[1]:
            raise RuntimeError("Please pass vector of len {}".format(self.vectors.shape[1]))

        if k not in self.vocabulary:
            self.vocabulary.add(k)
            self.vectors = np.vstack([self.vectors, v.reshape(1, -1)])
        else:
            self.vectors[self.vocabulary[k]] = v

    def __contains__(self, k):
        return k in self.vocabulary

    def __delitem__(self, k):
        """Remove the word and its vector from the embedding.

        Note:
         This operation costs \\theta(n). Be careful putting it in a loop.
        """
        index = self.vocabulary[k]
        del self.vocabulary[k]
        self.vectors = np.delete(self.vectors, index, 0)

    def __len__(self):
        return len(self.vocabulary)

    def __iter__(self):
        for w in self.vocabulary:
            yield w, self[w]

    @property
    def words(self):
        return self.vocabulary.words

    @property
    def shape(self):
        return self.vectors.shape

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError as e:
            return default

    def standardize_words(self, lower=False, clean_words=False, inplace=False):
        tw = self.transform_words(partial(standardize_string, lower=lower, clean_words=clean_words), inplace=inplace,
                                  lower=lower)

        if clean_words:
            tw = tw.transform_words(partial(lambda w: w.strip(" ")), inplace=inplace, lower=lower)
        return tw

    def transform_words(self, f, inplace=False, lower=False):
        """
        Transform words in vocabulary according to following strategy.
        Prefer shortest and most often occurring words- after transforming by some (lambda f) function.

        This allow eliminate noisy and wrong coded words.

        Strategy is implemented for all types of Vocabulary- they can be polymorphicaly extended.

        Parameters
        ----------
        f: lambda
            Function called on each word- for transformation it.

        inplace: bool, default: False
            Return new Embedding instance or modify existing

        lower: bool, default: False
            If true, will convert all words to lowercase

        Returns
        -------
        e: Embedding
        Instance of Embedding class with this same Vocabulary type as previous.
        """
        id_map = OrderedDict()
        word_count = len(self.vectors)
        # store max word length before f(w)- in corpora
        words_len = {}
        # store max occurrence count of word
        counts = {}
        is_vocab_generic = False

        curr_words = self.vocabulary.words
        curr_vec = self.vectors

        if isinstance(self.vocabulary, CountedVocabulary):
            _, counter_of_words = self.vocabulary.getstate()
        elif isinstance(self.vocabulary, OrderedVocabulary):
            # range in python3 is lazy
            counter_of_words = range(len(self.vocabulary.words) - 1, -1, -1)

        elif isinstance(self.vocabulary, Vocabulary):
            is_vocab_generic = True
            # if corpora contain lowercase version of word i- for case Vocabulary
            lowered_words = {}

            if lower:

                for w, v in zip(self.vocabulary.words, self.vectors):
                    wl = w.lower()
                    if wl == w:
                        lowered_words[wl] = v
                    elif wl != w and wl not in lowered_words:
                        lowered_words[wl] = v

                curr_words = list(lowered_words.keys())
                curr_vec = np.asanyarray(list(lowered_words.values()))

        else:
            raise NotImplementedError(
                'This kind of Vocabulary is not implemented in transform_words strategy and can not be matched')

        for id, w in enumerate(curr_words):

            fw = f(w)
            if len(fw) and fw not in id_map:
                id_map[fw] = id

                if not is_vocab_generic:
                    counts[fw] = counter_of_words[id]
                words_len[fw] = len(w)

                # overwrite
            elif len(fw) and fw in id_map:
                if not is_vocab_generic and counter_of_words[id] > counts[fw]:
                    id_map[fw] = id

                    counts[fw] = counter_of_words[id]
                    words_len[fw] = len(w)
                elif is_vocab_generic and len(w) < words_len[fw]:
                    # for generic Vocabulary
                    id_map[fw] = id

                    words_len[fw] = len(w)
                elif not is_vocab_generic and counter_of_words[id] == counts[fw] and len(w) < words_len[fw]:
                    id_map[fw] = id

                    counts[fw] = counter_of_words[id]
                    words_len[fw] = len(w)

                logger.warning("Overwriting {}".format(fw))

        if isinstance(self.vocabulary, CountedVocabulary):
            words_only = id_map.keys()
            vectors = curr_vec[[id_map[w] for w in words_only]]
            words = {w: counter_of_words[id_map[w]] for w in words_only}

        elif isinstance(self.vocabulary, OrderedVocabulary):
            words = sorted(id_map.keys(), key=lambda x: id_map[x])
            vectors = curr_vec[[id_map[w] for w in words]]

        elif isinstance(self.vocabulary, Vocabulary):
            words = sorted(id_map.keys(), key=lambda x: id_map[x])
            vectors = curr_vec[[id_map[w] for w in words]]

        logger.info("Transformed {} into {} words".format(word_count, len(words)))

        if inplace:
            self.vectors = vectors
            self.vocabulary = self.vocabulary.__class__(words)

            return self
        else:
            return Embedding(vectors=vectors, vocabulary=self.vocabulary.__class__(words))

    def most_frequent(self, k, inplace=False):
        """Only most frequent k words to be included in the embeddings."""

        assert isinstance(self.vocabulary, OrderedVocabulary), \
            "most_frequent can be called only on Embedding with OrderedVocabulary"

        vocabulary = self.vocabulary.most_frequent(k)
        vectors = np.asarray([self[w] for w in vocabulary])
        if inplace:
            self.vocabulary = vocabulary
            self.vectors = vectors
            return self
        return Embedding(vectors=vectors, vocabulary=vocabulary)

    def normalize_words(self, ord=2, inplace=False):
        """Normalize embeddings matrix row-wise.

        Parameters
        ----------
          ord: normalization order. Possible values {1, 2, 'inf', '-inf'}
        """
        if ord == 2:
            ord = None  # numpy uses this flag to indicate l2.
        vectors = self.vectors.T / np.linalg.norm(self.vectors, ord, axis=1)
        if inplace:
            self.vectors = vectors.T
            return self
        return Embedding(vectors=vectors.T, vocabulary=self.vocabulary)

    def nearest_neighbors(self, word, k=1, exclude=[], metric="cosine"):
        """
        Find nearest neighbor of given word

        Parameters
        ----------
          word: string or vector
            Query word or vector.

          k: int, default: 1
            Number of nearest neighbours to return.

          metric: string, default: 'cosine'
            Metric to use.

          exclude: list, default: []
            Words to omit in answer

        Returns
        -------
          n: list
            Nearest neighbors.
        """
        if isinstance(word, string_types):
            assert word in self, "Word not found in the vocabulary"
            v = self[word]
        else:
            v = word

        D = pairwise_distances(self.vectors, v.reshape(1, -1), metric=metric)

        if isinstance(word, string_types):
            D[self.vocabulary.word_id[word]] = D.max()

        for w in exclude:
            D[self.vocabulary.word_id[w]] = D.max()

        return [self.vocabulary.id_word[id] for id in D.argsort(axis=0).flatten()[0:k]]



class WordAnalogyBenchmark(DataSetBasedWordEmbeddingBenchmark):
    def __init__(self, test_set, method="3CosAdd", multiprocessing: bool = False, max_cpus=None):
        self.method = method
        self.max_cpus = max_cpus
        self.multiprocessing = multiprocessing
        super(WordAnalogyBenchmark, self).__init__(test_set)

    def method_name(self):
        return "wordanalogy"

    def _load_shipped_test_set(self, test_set):
        """"

        """
        path = get_shipped_test_set_path("word-analogy", test_set)
        return read_csv(path)

    def available_test_sets(self):
        return get_list_of_shipped_test_sets("word-analogy")

    def most_similar(self, calculated_vector, model: SeaQuBeWordEmbeddingsModel, topn=10):
        vocab_len = len(model.vocabs())

        distances = np.array([np.linalg.norm(calculated_vector - model.matrix()[i]) for i in range(vocab_len)])
        found_indecies = np.array(distances).argsort()[0:topn]

        return list(zip(np.array(model.vocabs())[found_indecies], distances[found_indecies]))

    def _nearest_neighbors(self, a, b, c, model: SeaQuBeWordEmbeddingsModel, exclude=[]):

        """
        See https://github.com/kudkudak/word-embeddings-benchmarks/blob/master/web/embedding.py
        Find nearest neighbor of given word
        Parameters
        ----------
          word: string or vector
            Query word or vector.
          k: int, default: 1
            Number of nearest neighbours to return.
          metric: string, default: 'cosine'
            Metric to use.
          exclude: list, default: []
            Words to omit in answer
        Returns
        -------
          n: list
            Nearest neighbors.
        """
        k = 10
        # exclude = []
        metric = "cosine"

        # if isinstance(word, str):
        #     assert word in vocabs, "Word not found in the vocabulary"
        #     v = w2v.wv[word]
        # else:  # otherwise it is a vector
        #     v = word
        v = a + b - c

        D = pairwise_distances(model.matrix(), v.reshape(1, -1), metric=metric)

        for w in exclude:
            D[self.word2index[w]] = D.max()

        return [self.index2word[id] for id in D.argsort(axis=0).flatten()[0:k]]


    def _nearest_neighbors_woembe(self, a, b, c, model: SeaQuBeWordEmbeddingsModel, exclude=[]):
        # original
        #return self.w.nearest_neighbors(self.w[b] - self.w[a] + self.w[c], exclude=[a, b, c])[0]
        # half adapted
        return self.w.nearest_neighbors(b - a + c, exclude=exclude)[0]

    def _3_cos_add(self, a, b, c, model: SeaQuBeWordEmbeddingsModel, exclude=[]):
        vocab_len = model.matrix().shape[0]
        res = []
        for i in range(vocab_len):
            d = model.matrix()[i]
            res.append(cosine(d, c) - cosine(d, a) + cosine(d, b))

        sorted_zip = list(zip(
            np.array(model.vocabs())[np.argsort(res)[::-1]],  # [::-1] is for maximizing
            np.sort(res)[::-1]
        ))

        sorted_top = sorted_zip[0:10]
        del sorted_zip
        gc.collect()
        return sorted_top

    def _vector_calc(self, a, b, c, model: SeaQuBeWordEmbeddingsModel, exclude=[]):
        calculated_wv = a - b + c
        return self.most_similar(calculated_wv, model)

    def _pair_dir(self, a, b, c, model: SeaQuBeWordEmbeddingsModel, exclude=[]):
        vocab_len = model.matrix().shape[0]
        res = []
        for i in range(vocab_len):
            d = model.matrix()[i]
            res.append(cosine(d - c, b - a))

        return list(zip(
            np.array(model.vocabs())[np.argsort(res)[::-1]],  # [::-1] is for maximizing
            np.sort(res)[::-1]
        ))[0:10]

    def _space_evolution(self, a, b, c, model: SeaQuBeWordEmbeddingsModel, exclude=[]):
        a_h = a / np.linalg.norm(a)
        b_h = b / np.linalg.norm(b)
        c_h = c / np.linalg.norm(c)

        return self._3_cos_add(a_h, b_h, c_h, model)

    def apply_on_testset_line(self, row):
        a, b, c = row.word1.lower(), row.word2.lower(), row.word3.lower()
        log.info(f"WordAnalogy of these relation:{a}:{b}::{c}:?")
        target = row.target.lower()

        log.info(f"WordAnalogy: target={target}")


        detected_targets = self.measure_method(self.model.wv[a], self.model.wv[b],self.model.wv[c], self.model,
                                        #a, b, c, self.model,

                                               exclude=[a, b, c])
        #row.word1, row.word2, row.word3

        log.info(f"WordAnalogy: detected_targets={detected_targets}")

        word = detected_targets[0][0]

        del detected_targets
        return int(word == target)

    def __call__(self, model: SeaQuBeWordEmbeddingsModel) -> BenchmarkScore:
        correct_hits = 0

        if self.method == "3CosAdd":
            self.measure_method = self._3_cos_add
        elif self.method == 'VectorCalc':
            self.measure_method = self._vector_calc
        elif self.method == 'PairDir':
            self.measure_method = self._pair_dir
        elif self.method == 'SpaceEvolution':
            self.measure_method = self._space_evolution
        elif self.method == "NearestNeighbors":

            #self.word2index = {word: i for i, word in enumerate(model.vocabs())}
            #self.index2word = {i: word for i, word in enumerate(model.vocabs())}
            self.w = Embedding(CountedVocabulary.from_vocabs(model.vocabs()), model.matrix())

            self.measure_method = self._nearest_neighbors_woembe
        else:
            raise ValueError(
                f"Argument `method` must be in one of [3CosAdd, VectorCalc, PairDir, SpaceEvolution, NearestNeighbors]")

        self.model = model

        # first filter dataset
        # all words need to be in the vocab list, otherwise it makes no sense
        filtered_rows = []
        for rowitem in progressbar(self.test_set.iterrows(), max_value=len(self.test_set)):
            _, row = rowitem

            if row.word1 in self.model.vocabs() and row.word2 in self.model.vocabs() and row.word3 in self.model.vocabs() and row.target in self.model.vocabs():
                filtered_rows.append(row)

        # then use filtered row for hard work
        prg = ProgressBar(max_value=len(filtered_rows))
        if self.multiprocessing:
            multi_wrapper = ForEach(self.apply_on_testset_line, max_cpus=self.max_cpus)
        else:
            def multi_wrapper(rows):
                for doc in rows:
                    yield self.apply_on_testset_line(doc)

        for correct_flag in multi_wrapper(filtered_rows):
            correct_hits += correct_flag
            prg.update(prg.value + 1)

        considered_lines = len(filtered_rows)

        if considered_lines == 0:
            return BenchmarkScore(0.0)

        return BenchmarkScore(correct_hits / considered_lines, {'matched_words': considered_lines,
                                                                'correct_hits': correct_hits})
