'''
Copyright (c) 2020 by Jason Wei and Kai Zou and Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: By Jason Wei, Kai Zou, Benjamin Manns
'''

import re
import random
# for the first time you use wordnet
    # import nltk
    # nltk.download('wordnet')
from nltk.corpus import wordnet

from seaqube.augmentation.base import MultiprocessingAugmentation
from seaqube.nlp.tools import tokenize_corpus
from seaqube.package_config import log


class EDAAugmentation(MultiprocessingAugmentation):
    """
    Algorithm provided by https://arxiv.org/pdf/1901.11196.pdf and https://github.com/jasonwei20/eda_nlp. It applies several very easy text augmentation methods, such as
    Synonym replacement, Random Swap, Random Insertion and Random Deletion

    # Easy data augmentation techniques for text classification
    # Jason Wei and Kai Zou
    # cleaning up text
    """
    def input_type(self):
        """
        Which return type is supported
        Returns: doc or text
        """
        return "text"

    def __init__(self, alpha_sr: float = 0.1, alpha_ri: float = 0.1, alpha_rs: float = 0.1, p_rd: float = 0.1,
                 num_aug: int = 9, max_length: int = 100, remove_duplicates: bool = False, multiprocess: bool = True,
                 seed: int = None):
        """
        Args:
            alpha_sr: probability of synonym replacement
            alpha_ri: probability of random insertion
            alpha_rs: probability of random swap
            p_rd: probability of random deletion
            num_aug: how many single augmentations should performed
            max_length: cut the produced text at a limit to prevent overflow
            remove_duplicates: remove after augmentation for duplicates
            multiprocess: if augmentation class implements the multiprocessing call, then it can be turn off again with
                    this flag, most for testing purpose
            seed: fix the randomness with a seed for testing purpose
        """

        self.alpha_sr = alpha_sr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.p_rd = p_rd
        self.num_aug = num_aug
        self.max_length = max_length
        self.remove_duplicates = remove_duplicates
        self.seed = seed
        self.random = random.Random()
        self.multiprocess = multiprocess

        if self.seed is not None:
            self.random.seed(self.seed)

        # stop words list
        self.stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
                      'ours', 'ourselves', 'you', 'your', 'yours',
                      'yourself', 'yourselves', 'he', 'him', 'his',
                      'himself', 'she', 'her', 'hers', 'herself',
                      'it', 'its', 'itself', 'they', 'them', 'their',
                      'theirs', 'themselves', 'what', 'which', 'who',
                      'whom', 'this', 'that', 'these', 'those', 'am',
                      'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'having', 'do', 'does', 'did',
                      'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                      'because', 'as', 'until', 'while', 'of', 'at',
                      'by', 'for', 'with', 'about', 'against', 'between',
                      'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'to', 'from', 'up', 'down', 'in',
                      'out', 'on', 'off', 'over', 'under', 'again',
                      'further', 'then', 'once', 'here', 'there', 'when',
                      'where', 'why', 'how', 'all', 'any', 'both', 'each',
                      'few', 'more', 'most', 'other', 'some', 'such', 'no',
                      'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                      'very', 's', 't', 'can', 'will', 'just', 'don',
                      'should', 'now', '']

    def get_config(self):
        """
        Gives a dict with all relevant variables the object can recreated with (init parameters)
        Returns: dict of object config

        """
        return dict(alpha_sr=self.alpha_sr, alpha_ri=self.alpha_ri, alpha_rs=self.alpha_rs, p_rd=self.p_rd, num_aug=self.num_aug,
                    max_length=self.max_length, remove_duplicates=self.remove_duplicates, seed=self.seed, class_name=str(self))

    def augmentation_implementation(self, sentence):
        if len(sentence.strip()) == 0:
            return []

        try:
            eda = self.eda(sentence, self.alpha_sr, self.alpha_ri, self.alpha_rs, self.p_rd, self.num_aug)
            return tokenize_corpus(eda, verbose=False)[0: self.max_length]
        except ValueError:
            return []

    def shortname(self):
        return "eda"

    def get_only_chars(self, line):
        clean_line = ""

        line = line.replace("’", "")
        line = line.replace("'", "")
        line = line.replace("-", " ")  # replace hyphens with spaces
        line = line.replace("\t", " ")
        line = line.replace("\n", " ")
        line = line.lower()

        for char in line:
            if char in 'qwertyuiopasdfghjklzxcvbnm ':
                clean_line += char
            else:
                clean_line += ' '

        clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
        if clean_line[0] == ' ':
            clean_line = clean_line[1:]
        return clean_line

    ########################################################################
    # Synonym replacement
    # Replace n words in the sentence with synonyms from wordnet
    ########################################################################



    def synonym_replacement(self, words, n):
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stop_words]))
        self.random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = self.random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                log.debug(f"{self.__class__.__name__}: replaced={random_word} with {synonym}")
                num_replaced += 1
            if num_replaced >= n:  # only replace up to n words
                break

        # this is stupid but we need it, trust me
        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')

        return new_words

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym)
        if word in synonyms:
            synonyms.remove(word)

        synonyms_cleaned = list(synonyms)
        return synonyms_cleaned

    ########################################################################
    # Random deletion
    # Randomly delete words from the sentence with probability p
    ########################################################################

    def random_deletion(self, words, p):
        # obviously, if there's only one word, don't delete it
        if len(words) == 1:
            return words

        # randomly delete words with probability p
        new_words = []
        for word in words:
            r = self.random.uniform(0, 1)
            if r > p:
                new_words.append(word)

        # if you end up deleting all words, just return a random word
        if len(new_words) == 0:
            rand_int = self.random.randint(0, len(words) - 1)
            return [words[rand_int]]

        return new_words

    ########################################################################
    # Random swap
    # Randomly swap two words in the sentence n times
    ########################################################################

    def random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    def swap_word(self, new_words):
        random_idx_1 = self.random.randint(0, len(new_words) - 1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = self.random.randint(0, len(new_words) - 1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
        return new_words

    ########################################################################
    # Random insertion
    # Randomly insert n words into the sentence
    ########################################################################

    def random_insertion(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, new_words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = new_words[self.random.randint(0, len(new_words) - 1)]
            synonyms = self.get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = synonyms[0]
        random_idx = self.random.randint(0, len(new_words) - 1)
        new_words.insert(random_idx, random_synonym)

    ########################################################################
    # main data augmentation function
    ########################################################################

    def eda(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
        sentence = self.get_only_chars(sentence)
        words = sentence.split(' ')
        words = [word for word in words if word is not '']
        num_words = len(words)

        augmented_sentences = []
        num_new_per_technique = int(num_aug / 4) + 1
        n_sr = max(1, int(alpha_sr * num_words))
        n_ri = max(1, int(alpha_ri * num_words))
        n_rs = max(1, int(alpha_rs * num_words))

        # sr
        for _ in range(num_new_per_technique):
            a_words = self.synonym_replacement(words, n_sr)
            augmented_sentences.append(' '.join(a_words))

        # ri
        for _ in range(num_new_per_technique):
            a_words = self.random_insertion(words, n_ri)
            augmented_sentences.append(' '.join(a_words))

        # rs
        for _ in range(num_new_per_technique):
            a_words = self.random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words))

        # rd
        for _ in range(num_new_per_technique):
            a_words = self.random_deletion(words, p_rd)
            augmented_sentences.append(' '.join(a_words))

        augmented_sentences = [self.get_only_chars(sentence) for sentence in augmented_sentences]
        self.random.shuffle(augmented_sentences)

        # trim so that we have the desired number of augmented sentences
        if num_aug >= 1:
            augmented_sentences = augmented_sentences[:num_aug]
        else:
            keep_prob = num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if self.random.uniform(0, 1) < keep_prob]

        # append the original sentence
        augmented_sentences.append(sentence)

        return augmented_sentences

