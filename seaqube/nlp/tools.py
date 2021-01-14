"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns

The nlp tool-kit which makes it possible to do text and doc loading, saving, processing in one-liners.

"""

from collections import Counter
from copy import deepcopy
from collections import Iterable
from seaqube.nlp.types import SeaQuBeWordEmbeddingsModelGensim, BackgroundScheduler, SeaQuBeWordEmbeddingsModelC2V
from seaqube.package_config import log
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from progressbar import progressbar

anti_tokenizer = TreebankWordDetokenizer()

def word_count_list(double_list):
    """
    Create a word count list of a corpus in doc format.
    """
    word_count = Counter()

    for sentence in double_list:
        for token in sentence:
            word_count[token] += 1  # Equivalently, token.text
    return word_count


def gensim_we_model_to_custom_we_model(gensim_model):
    return SeaQuBeWordEmbeddingsModelGensim(gensim_model)


def c2_we_model_to_custom_we_model(c2v_model):
    return SeaQuBeWordEmbeddingsModelC2V(c2v_model)


class DelayedListToJSONWriter:
    """
    @deprecated
    Write buffer of corpus down, while not preserving gig's of text in RAM.
    """
    def __init__(self, file_path, buffer_size=10):
        self.path = file_path
        self.buffer = []
        self.buffer_size = buffer_size
        self._in_write_mode = False
        self.background_scheduler = BackgroundScheduler()
        self.background_scheduler.start()

        self.background_scheduler.local_scheduler.every(5).minutes.do(self.__save_on_disk)

    def finalize(self):
        log.debug(f"{self.__class__.__name__}: finalize is entered now")
        self.background_scheduler.stop()
        self.buffer_size = 0
        self.__save_on_disk()
        self.background_scheduler.join()

    def __save_on_disk(self):
        if not self._in_write_mode:
            self._in_write_mode = True
            log.debug("Check, save on disk")
            data = []
            if len(self.buffer) >= self.buffer_size:
                buffer_copy = deepcopy(self.buffer)
                self.buffer = []
                ## we make a fake json, but it is efficient
                #if exists(self.path):
                #   data = load_json(self.path)
                #data += self.buffer
                #self.buffer = []
                #save_json(data, self.path)
                with open(self.path, "a") as f:
                    f.write(json.dumps(buffer_copy) + "\n")


                log.debug("DelayedListToJSONWriter: write to file")
            self._in_write_mode = False


    def add(self, elem):
        self.buffer.append(elem)




def tokenize_corpus(sentences: Iterable, verbose=True):
    """
    Tokenize a list of texts.
    """
    if verbose:
        sentences = progressbar(sentences)
    return [[token.lower() for token in word_tokenize(sentence) if token.isspace() is False] for sentence in sentences]


def sentenceize_corpus(token_list: list, verbose=True):
    """
    Reverse tokenization and transform a corpus again as a list of texts.
    """
    if verbose:
        token_list = progressbar(token_list)
    return [anti_tokenizer.detokenize(tokens) for tokens in token_list]


def unique_2d_list(list_: list):
    """
    Only keep those lists inside the big list which are unique.
    Args:
        list_: list of list where second dimension can contain duplicates
    Returns:
    """
    return list(map(lambda x: list(x), set(map(lambda x: tuple(x), list_))))
