"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

import random

from seaqube.augmentation.base import MultiprocessingAugmentation


class QwertyAugmentation(MultiprocessingAugmentation):
    """
    Based on an idea from https://amitness.com/2020/05/data-augmentation-for-nlp/ textes are injected with keyboard typos
    """

    def __init__(self, replace_rate: float = 0.01, caps_change: float = 0.1, max_length: int = 100,
                 remove_duplicates: bool = False, multiprocess: bool = True, seed: int = None):
        """"
        :param float replace_rate: propability that a char around the given is pressed
        :param float caps_change: propability that a upper/lower changes
        :param int max_length: how many texts should produced
        :param bool remove_duplicates: remove after augmentation for duplicates
        :param multiprocess: if augmentation class implements the multiprocessing call, then it can be turn off again with
                    this flag, most for testing purpose
        :param int seed: fix the randomness with a seed for testing purpose
        """
        self.replace_rate = replace_rate
        self.caps_change = caps_change
        self.max_length = max_length
        self.remove_duplicates = remove_duplicates
        self.r = random.Random()
        self.seed = seed
        self.multiprocess = multiprocess

        if self.seed is not None:
            self.r.seed(self.seed)

    def get_config(self):
        """
        Gives a dict with all relevant variables the object can recreated with (init parameters)
        Returns: dict of object config

        """
        return dict(replace_rate=self.replace_rate, caps_change=self.caps_change, max_length=self.max_length,
                    remove_duplicates=self.remove_duplicates, seed=self.seed, class_name=str(self))

    def input_type(self):
        """
        Which return type is supported
        Returns: doc or text
        """
        return "doc"

    def augmentation_implementation(self, doc):
        """
        Run the qwerty on every term
        Returns: doc
        """
        return [self.__qwerty(doc) for _ in range(self.max_length)]

    def __qwerty(self, doc):
        """
        Add typos to words, given a list of words
        Returns: doc
        """
        doc = map(lambda x: list(x), doc)

        doc_new = []
        for word_index, word in enumerate(doc):
            for char_index, char in enumerate(word):
                #if char in self.__table().keys():
                try:
                    if (1 - self.replace_rate) < self.r.random():
                        typ = ['neighbour', 'capsed'][int((1 - self.caps_change) < self.r.random())]
                        word[char_index] = self.r.sample(self.__table()[char][typ], 1)[0]
                except KeyError:
                    pass

            doc_new.append("".join(word))

        return doc_new

    def shortname(self):
        return "qwerty"

    def __table(self):
        return {'~': {'neighbour': ['1', 'q'], 'capsed': ['Q', '!', '`']}, '1': {'neighbour': ['~', 'w', 'q', '2'], 'capsed': ['Q', '@', '!', '`', 'W']}, '2': {'neighbour': ['3', 'q', 'e', 'w', '1'], 'capsed': ['Q', '@', '!', 'E', '#', 'W']}, '3': {'neighbour': ['e', '2', '4', 'w', 'r'], 'capsed': ['@', 'E', '$', 'R', '#', 'W']}, '4': {'neighbour': ['3', '5', 't', 'e', 'r'], 'capsed': ['E', 'T', '$', 'R', '#', '%']}, '5': {'neighbour': ['6', 'y', 't', '4', 'r'], 'capsed': ['T', '$', 'R', '^', 'Y', '%']}, '6': {'neighbour': ['5', '7', 'y', 't', 'u'], 'capsed': ['T', '&', '^', 'Y', 'U', '%']}, '7': {'neighbour': ['6', '8', 'y', 'i', 'u'], 'capsed': ['Y', 'U', '&', '^', '*', 'I']}, '8': {'neighbour': ['o', '7', 'i', '9', 'u'], 'capsed': ['I', 'O', '&', '(', '*', 'U']}, '9': {'neighbour': ['o', '8', 'i', '0', 'p'], 'capsed': ['O', '(', 'P', ')', '*', 'I']}, '0': {'neighbour': ['o', '[', '9', 'p', '-'], 'capsed': ['O', '(', 'P', '{', ')', '_']}, '-': {'neighbour': [']', '[', '0', '=', 'p'], 'capsed': ['P', '{', '+', '_', ')', '}']}, '=': {'neighbour': [']', '[', '-'], 'capsed': ['{', '}', '_', '+']}, '`': {'neighbour': ['Q', '!'], 'capsed': ['~', '1', 'q']}, '!': {'neighbour': ['Q', '@', '`', 'W'], 'capsed': ['~', 'q', '2', 'w', '1']}, '@': {'neighbour': ['Q', '!', 'E', 'W', '#'], 'capsed': ['3', 'q', 'e', 'w', '2', '1']}, '#': {'neighbour': ['@', 'E', '$', 'R', 'W'], 'capsed': ['3', 'e', '2', '4', 'w', 'r']}, '$': {'neighbour': ['E', 'T', 'R', '#', '%'], 'capsed': ['3', '5', 't', 'e', '4', 'r']}, '%': {'neighbour': ['T', '$', 'R', '^', 'Y'], 'capsed': ['6', '5', 'y', 't', '4', 'r']}, '^': {'neighbour': ['T', '&', 'Y', 'U', '%'], 'capsed': ['6', '5', '7', 'y', 't', 'u']}, '&': {'neighbour': ['Y', 'I', '^', '*', 'U'], 'capsed': ['6', '7', '8', 'y', 'i', 'u']}, '*': {'neighbour': ['O', '&', '(', 'I', 'U'], 'capsed': ['o', '7', '8', 'i', '9', 'u']}, '(': {'neighbour': ['O', 'P', ')', '*', 'I'], 'capsed': ['o', '8', 'i', '9', '0', 'p']}, ')': {'neighbour': ['O', '(', 'P', '{', '_'], 'capsed': ['o', '[', '9', '0', 'p', '-']}, '_': {'neighbour': ['P', '{', '+', ')', '}'], 'capsed': [']', '[', '0', '=', 'p', '-']}, '+': {'neighbour': ['{', '}', '_'], 'capsed': [']', '[', '-', '=']}, 'q': {'neighbour': ['~', 'a', 's', '2', 'w', '1'], 'capsed': ['A', '@', 'Q', '!', 'S', '`', 'W']}, 'w': {'neighbour': ['3', 'a', 's', 'e', 'q', '2', '1', 'd'], 'capsed': ['A', 'Q', '!', '@', 'S', 'E', '#', 'W', 'D']}, 'e': {'neighbour': ['3', 'f', 's', '2', '4', 'w', 'd', 'r'], 'capsed': ['@', 'S', 'E', 'F', 'R', '$', 'W', '#', 'D']}, 'r': {'neighbour': ['g', '3', '5', 't', 'f', 'e', '4', 'd'], 'capsed': ['E', 'F', 'T', 'R', '$', '#', 'G', 'D', '%']}, 't': {'neighbour': ['6', 'g', '5', 'h', 'y', 'f', '4', 'r'], 'capsed': ['H', 'F', 'T', 'R', '$', '^', 'G', 'Y', '%']}, 'y': {'neighbour': ['6', 'g', 'j', '5', 'h', '7', 't', 'u'], 'capsed': ['H', 'T', '&', '^', 'J', 'G', 'Y', 'U', '%']}, 'u': {'neighbour': ['6', 'j', 'h', '7', '8', 'y', 'i', 'k'], 'capsed': ['Y', 'I', 'H', '&', 'K', '^', 'J', '*', 'U']}, 'i': {'neighbour': ['j', 'o', '7', '8', '9', 'l', 'u', 'k'], 'capsed': ['O', '&', 'L', 'K', '*', '(', 'J', 'I', 'U']}, 'o': {'neighbour': ['8', 'i', '9', '0', 'l', 'p', 'k', ';'], 'capsed': ['O', 'K', 'L', '*', '(', 'P', ')', ':', 'I']}, 'p': {'neighbour': [';', 'o', "'", '[', '9', '0', 'l', '-'], 'capsed': ['O', 'L', '"', '(', 'P', ')', '{', ':', '_']}, '[': {'neighbour': [']', ';', "'", '0', '=', 'p', '-'], 'capsed': [')', '"', 'P', '+', '_', '{', ':', '}']}, ']': {'neighbour': ['[', "'", '-', '='], 'capsed': ['}', '"', '+', '{', '_']}, '\\': {'neighbour': [']', '='], 'capsed': ['}', '+']}, 'Q': {'neighbour': ['A', 'S', '@', '!', '`', 'W'], 'capsed': ['~', 'a', 's', 'q', '2', 'w', '1']}, 'W': {'neighbour': ['A', '@', 'Q', '!', 'S', 'E', '#', 'D'], 'capsed': ['3', '2', 'a', 'e', 'q', 's', 'w', '1', 'd']}, 'E': {'neighbour': ['@', 'S', 'F', 'R', '$', '#', 'W', 'D'], 'capsed': ['3', 'f', 's', 'e', '2', '4', 'w', 'd', 'r']}, 'R': {'neighbour': ['E', 'F', 'T', '$', '#', 'G', 'D', '%'], 'capsed': ['g', '3', '5', 't', 'f', 'e', '4', 'd', 'r']}, 'T': {'neighbour': ['H', 'F', 'R', '$', '^', 'G', 'Y', '%'], 'capsed': ['6', 'g', '5', 'h', 'y', 't', 'f', '4', 'r']}, 'Y': {'neighbour': ['H', 'T', '&', '^', 'J', 'G', 'U', '%'], 'capsed': ['6', 'g', 'j', '5', 'h', '7', 'y', 't', 'u']}, 'U': {'neighbour': ['Y', 'H', '&', 'K', '^', 'J', '*', 'I'], 'capsed': ['6', 'j', 'h', '7', '8', 'y', 'i', 'u', 'k']}, 'I': {'neighbour': ['O', '&', 'L', 'K', '(', 'J', '*', 'U'], 'capsed': ['j', 'o', '7', '8', 'i', '9', 'l', 'u', 'k']}, 'O': {'neighbour': ['K', 'L', '(', 'P', ')', ':', '*', 'I'], 'capsed': ['o', '8', 'i', '9', '0', 'l', 'p', 'k', ';']}, 'P': {'neighbour': ['O', 'L', '"', '(', ')', '{', ':', '_'], 'capsed': [';', 'o', "'", '[', '9', '0', 'l', 'p', '-']}, '{': {'neighbour': ['}', '"', 'P', '+', ')', ':', '_'], 'capsed': [']', ';', "'", '[', '0', '=', 'p', '-']}, '}': {'neighbour': ['{', '"', '_', '+'], 'capsed': [']', "'", '[', '=', '-']}, '|': {'neighbour': ['}', '+'], 'capsed': [']', '=']}, 'a': {'neighbour': ['z', 'w', 's', 'q'], 'capsed': ['A', 'S', 'Q', 'Z', 'W']}, 's': {'neighbour': ['x', 'a', 'e', 'q', 'z', 'w', 'd'], 'capsed': ['A', 'S', 'Q', 'E', 'X', 'Z', 'W', 'D']}, 'd': {'neighbour': ['x', 'c', 'f', 's', 'e', 'z', 'w', 'r'], 'capsed': ['S', 'E', 'F', 'X', 'R', 'Z', 'C', 'W', 'D']}, 'f': {'neighbour': ['g', 'x', 'c', 't', 'e', 'd', 'r', 'v'], 'capsed': ['E', 'F', 'T', 'X', 'R', 'C', 'G', 'D', 'V']}, 'g': {'neighbour': ['h', 'b', 'y', 'c', 't', 'f', 'r', 'v'], 'capsed': ['H', 'F', 'T', 'R', 'C', 'G', 'B', 'Y', 'V']}, 'h': {'neighbour': ['g', 'j', 'b', 'n', 'y', 't', 'u', 'v'], 'capsed': ['N', 'H', 'T', 'J', 'G', 'B', 'Y', 'U', 'V']}, 'j': {'neighbour': ['h', 'b', 'n', 'y', 'i', 'm', 'u', 'k'], 'capsed': ['N', 'U', 'H', 'M', 'K', 'J', 'B', 'Y', 'I']}, 'k': {'neighbour': ['<', 'j', 'o', 'n', 'i', 'm', 'l', 'u'], 'capsed': ['N', 'O', 'M', 'K', 'L', 'J', ',', 'I', 'U']}, 'l': {'neighbour': ['<', 'o', 'i', '>', 'm', 'p', 'k', ';'], 'capsed': ['.', ',', 'O', 'M', 'L', 'K', 'P', ':', 'I']}, ';': {'neighbour': ['<', 'o', "'", '[', '>', 'l', '?', 'p'], 'capsed': ['.', ',', 'O', 'L', '"', '/', 'P', '{', ':']}, "'": {'neighbour': [']', '[', '>', '?', 'p', ';'], 'capsed': ['.', '"', '/', 'P', '{', ':', '}']}, 'A': {'neighbour': ['Q', 'S', 'Z', 'W'], 'capsed': ['a', 's', 'q', 'z', 'w']}, 'S': {'neighbour': ['A', 'Q', 'E', 'X', 'Z', 'W', 'D'], 'capsed': ['x', 'a', 's', 'e', 'q', 'z', 'w', 'd']}, 'D': {'neighbour': ['S', 'E', 'F', 'X', 'R', 'Z', 'C', 'W'], 'capsed': ['x', 'c', 'f', 's', 'e', 'z', 'w', 'd', 'r']}, 'F': {'neighbour': ['E', 'T', 'X', 'R', 'C', 'G', 'D', 'V'], 'capsed': ['g', 'x', 'c', 'f', 't', 'e', 'd', 'r', 'v']}, 'G': {'neighbour': ['H', 'F', 'T', 'R', 'C', 'B', 'Y', 'V'], 'capsed': ['g', 'h', 'b', 'y', 'c', 'f', 't', 'r', 'v']}, 'H': {'neighbour': ['N', 'T', 'J', 'G', 'B', 'Y', 'U', 'V'], 'capsed': ['g', 'j', 'h', 'b', 'n', 'y', 't', 'u', 'v']}, 'J': {'neighbour': ['N', 'I', 'H', 'M', 'K', 'B', 'Y', 'U'], 'capsed': ['j', 'h', 'b', 'n', 'y', 'i', 'm', 'u', 'k']}, 'K': {'neighbour': ['N', 'O', 'M', 'L', 'J', ',', 'I', 'U'], 'capsed': ['<', 'j', 'o', 'n', 'i', 'm', 'l', 'u', 'k']}, 'L': {'neighbour': ['.', ',', 'O', 'M', 'K', 'P', ':', 'I'], 'capsed': ['<', 'o', 'i', '>', 'm', 'l', 'p', 'k', ';']}, ':': {'neighbour': ['.', 'O', 'L', '"', '/', 'P', '{', ','], 'capsed': ['<', 'o', "'", '[', '>', 'l', '?', 'p', ';']}, '"': {'neighbour': ['.', '/', 'P', '{', ':', '}'], 'capsed': [']', "'", '[', '>', '?', 'p', ';']}, 'z': {'neighbour': ['a', 's', 'x', 'd'], 'capsed': ['A', 'S', 'X', 'Z', 'D']}, 'x': {'neighbour': ['c', 'f', 's', 'z', 'd'], 'capsed': ['S', 'F', 'X', 'C', 'Z', 'D']}, 'c': {'neighbour': ['g', 'x', 'f', 'd', 'v'], 'capsed': ['F', 'X', 'C', 'G', 'D', 'V']}, 'v': {'neighbour': ['g', 'h', 'b', 'c', 'f'], 'capsed': ['H', 'F', 'C', 'G', 'B', 'V']}, 'b': {'neighbour': ['g', 'j', 'h', 'n', 'v'], 'capsed': ['N', 'H', 'J', 'G', 'B', 'V']}, 'n': {'neighbour': ['j', 'h', 'b', 'm', 'k'], 'capsed': ['N', 'H', 'M', 'K', 'J', 'B']}, 'm': {'neighbour': ['<', 'j', 'n', 'l', 'k'], 'capsed': ['N', 'M', 'L', 'K', 'J', ',']}, '<': {'neighbour': ['>', 'm', 'l', 'k', ';'], 'capsed': ['.', 'M', ':', 'K', 'L', ',']}, '>': {'neighbour': ['<', "'", 'l', '?', ';'], 'capsed': ['.', ':', 'L', '"', '/', ',']}, '?': {'neighbour': ['>', ';', "'"], 'capsed': ['/', '"', ':', '.']}, 'Z': {'neighbour': ['A', 'S', 'D', 'X'], 'capsed': ['x', 'a', 's', 'z', 'd']}, 'X': {'neighbour': ['S', 'F', 'C', 'Z', 'D'], 'capsed': ['x', 'c', 'f', 's', 'z', 'd']}, 'C': {'neighbour': ['F', 'X', 'G', 'D', 'V'], 'capsed': ['g', 'x', 'c', 'f', 'd', 'v']}, 'V': {'neighbour': ['H', 'F', 'C', 'G', 'B'], 'capsed': ['g', 'h', 'b', 'c', 'f', 'v']}, 'B': {'neighbour': ['N', 'H', 'J', 'G', 'V'], 'capsed': ['g', 'j', 'b', 'h', 'n', 'v']}, 'N': {'neighbour': ['H', 'M', 'K', 'J', 'B'], 'capsed': ['j', 'h', 'b', 'n', 'm', 'k']}, 'M': {'neighbour': ['N', 'K', 'L', 'J', ','], 'capsed': ['<', 'j', 'n', 'm', 'l', 'k']}, ',': {'neighbour': ['.', 'M', 'L', 'K', ':'], 'capsed': ['<', '>', 'm', 'l', 'k', ';']}, '.': {'neighbour': [':', 'L', '"', '/', ','], 'capsed': ['<', "'", '>', 'l', '?', ';']}, '/': {'neighbour': ['"', ':', '.'], 'capsed': ['>', "'", ';', '?']}}
