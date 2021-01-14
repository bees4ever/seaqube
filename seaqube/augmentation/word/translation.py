"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""


import copy
import time

from googletrans import Translator

from seaqube.augmentation.base import SingleprocessingAugmentation
from seaqube.nlp.tools import tokenize_corpus
from seaqube.package_config import log


class TranslationAugmentation(SingleprocessingAugmentation):
    """
    Based on the idea from the author (Adams Wei Yu and David Dohan and Minh-Thang Luong and Rui Zhao and Kai Chen and
    Mohammad Norouzi and Quoc V. Le) of the paper
    "QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension"

    The idea is to translate a text into one or several language and translate it then back to the original one. The
    idea behind this, is to keep the content of a text but change it surface.

    The translator engine still is google translate, as long as it works


    @misc{yu2018qanet,
    title={QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension},
    author={Adams Wei Yu and David Dohan and Minh-Thang Luong and Rui Zhao and Kai Chen and Mohammad Norouzi and Quoc V. Le},
    year={2018},
    eprint={1804.09541},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
    }
    """

    def __init__(self, base_lang='en', max_length: int = 100, remove_duplicates: bool = False, timeout: int = None,
                 multiprocess: bool = True, seed: int = None):
        """
        Set up the translator for a given start / base language, default is en
        Args:
            base_lang: from where to and back translate and
            max_length: cut the produced text at a limit to prevent overflow
            remove_duplicates: remove after augmentation for duplicates
            timeout: number in seconds to wait until to fire next translation
            multiprocess: if augmentation class implements the multiprocessing call, then it can be turn off again with
                    this flag, most for testing purpose
            seed: fix the randomness with a seed for testing purpose
        """
        super(TranslationAugmentation, self).__init__()
        self.translator = Translator()
        self.base_lang = base_lang
        self.max_length = max_length
        self.remove_duplicates = remove_duplicates
        self.multiprocess = multiprocess
        self.seed = seed
        self.timeout = timeout
        self.last_call = 0.0



    def get_config(self):
        """
        Gives a dict with all relevant variables the object can recreated with (init parameters)
        Returns: dict of object config

        """
        return dict(base_lang=self.base_lang, max_length=self.max_length,
                    remove_duplicates=self.remove_duplicates, seed=self.seed, class_name=str(self))

    def shortname(self):
        return "googletranslate"

    def input_type(self):
        """
        Which return type is supported
        Returns: doc or text
        """
        return "text"

    def augmentation_implementation(self, sentence):
        return self.translate_doc(sentence)

    def __handle_timeout(self):
        if self.timeout is None:
            return

        diff = time.time() - self.last_call - self.timeout
        if diff < 0:
            time.sleep(abs(diff))

        self.last_call = time.time()
        return

    def translate_doc(self, text):
        translation_pipelines = [
            ['fr'],
            ['de'],
            ['sv'],
            ['de', 'fr'],
            ['ja'],
            ['la'],
            ['ko'],
            ['nl']
        ]

        texts = []

        for translation_pipeline in translation_pipelines:
            translation_pipeline = [self.base_lang] + translation_pipeline + [self.base_lang]
            tmp_text = copy.deepcopy(text)

            for i, lang in enumerate(translation_pipeline):
                next_lang = translation_pipeline[i+1]
                try:
                    self.__handle_timeout()
                    tmp_text = self.translator.translate(tmp_text, dest=next_lang, src=lang).text
                except Exception:
                    log.info("Some translation did not work, we try it later again")
                if next_lang == self.base_lang:
                    # Chain is finished
                    break
            texts.append(tmp_text)

        return tokenize_corpus(texts[0: self.max_length], verbose=False)
