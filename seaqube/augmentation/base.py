"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

from abc import abstractmethod
from typing import List
from progressbar import progressbar, ProgressBar

from seaqube.augmentation.reduction._reduction import BaseReduction
from seaqube.nlp.tools import tokenize_corpus, sentenceize_corpus, unique_2d_list
from seaqube.package_config import log
from seaqube.tools.io import DummyWriter
from seaqube.tools.types import Configable, Writeable
from seaqube.tools.umproc import ForEach


class BaseAugmentation(Configable):
    """Abstract class how implements augmentation procedure of a string, i.e. a human readable sentence or a
    already parsed and tokenized list of string, i.e. document.

    The sentence might be 'The quick brown fox jumps over the lazy dog'.
    The equivalent doc is: ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
    """

    def __init__(self, max_length: int = 100, remove_duplicates: bool = False, multiprocess: bool = True, seed: int = None):
        """
        A abstract constructor definition, every augmentation method needs to implement
        Args:
            max_length: cut the produced text at a limit to prevent overflow
            remove_duplicates: remove after augmentation for duplicates
            multiprocess: if augmentation class implements the multiprocessing call, then it can be turn off again with
                    this flag, most for testing purpose
            seed: fix the randomness with a seed for testing purpose
        """
        self.max_length = max_length
        self.remove_duplicates = remove_duplicates
        self.seed = seed

    def __str__(self):
        return str(self.__class__.__name__)

    def __repr__(self):
        return str(self)

    def doc_augment(self, **kwargs):
        """
        Perform a single augmentation based on the initialized augmentator. Choose either the `text` or `doc` parameter.
        :param str text: The sentence on which the augmentation should run
        :param list doc: The tokenized sentence, called doc on which the augmentation should run
        :return: list: List of doc, i.e. tokenized corpus
        """
        doc = kwargs.pop("doc", None)
        text = kwargs.pop("text", None)

        if (doc is None and text is None) or (doc is not None and text is not None):
            raise ValueError("Exact only one parameter of doc and text can be used")

        if doc is not None and isinstance(doc[0], list):
            raise ValueError("Proving a doc parameter, be sure to use a one-dimension list, for example ['the', 'brown', 'fox', 'jumps']")

        if self.input_type() == "doc" and text is not None:
            return self.augmentation_implementation(tokenize_corpus([text], verbose=False)[0])
        elif self.input_type() == "doc" and doc is not None:
            return self.augmentation_implementation(doc)
        if self.input_type() == "text" and text is not None:
            return self.augmentation_implementation(text)
        if self.input_type() == "text" and doc is not None:
            return self.augmentation_implementation(sentenceize_corpus([doc], verbose=False)[0])

    def augment(self, text):
        """
        A wrapper of doc_augment to integrate SeaQuBe Augmentation Classes smoothly into NLPAUG
        """
        return [sentenceize_corpus([doc])[0] for doc in self.doc_augment(text=text)]

    @abstractmethod
    def input_type(self):
        """
        Defines the type which mode the augmentation method supports
        :return str: doc or text or corpus
        """
        pass

    @abstractmethod
    def augmentation_implementation(self, document):
        pass

    @abstractmethod
    def shortname(self):
        """
        Each class has a name under which it is available
        :return str: shortname of augmenter
        """
        pass

    @abstractmethod
    def __call__(self, corpus):
        pass


class MultiprocessingAugmentation(BaseAugmentation):
    """
    The base implementation for augmentation which can run in multiprocessing
    """
    def __multi_wrapper_doc_augment(self, doc):
        return self.doc_augment(doc=doc)

    def __call__(self, corpus):
        if self.multiprocess:
            multi_wrapper = ForEach(self.__multi_wrapper_doc_augment)

            prg = ProgressBar(max_value=len(corpus))

            multi_wrapper.start()

            augmented_corpus = []
            for augmented_doc in multi_wrapper(corpus):
                augmented_corpus += augmented_doc
                prg.update(prg.value + 1)

            if self.remove_duplicates:
                augmented_corpus = unique_2d_list(augmented_corpus)

            return augmented_corpus

        else:
            augmented_corpus = []
            for doc in progressbar(corpus):
                augmented_corpus += self.doc_augment(doc=doc)

            if self.remove_duplicates:
                augmented_corpus = unique_2d_list(augmented_corpus)

            return augmented_corpus


class SingleprocessingAugmentation(BaseAugmentation):
    """
    The base implementation for augmentation which simply run in single processing
    """
    def __call__(self, corpus):

        augmented_corpus = []
        for doc in progressbar(corpus):
            augmented_corpus += self.doc_augment(doc=doc)

        if self.remove_duplicates:
            augmented_corpus = unique_2d_list(augmented_corpus)

        return augmented_corpus


class AugmentationStreamer:
    """
    The base implementation is doing augmentation streaming. This class sets up a stream like operation of a list of
    augmentation as well as a list of reduction classes.
    """
    def __init__(self, augmentations: List[BaseAugmentation], reduction_chain: List[BaseReduction] = [],
                 writer: Writeable = None):
        self.augmentations = augmentations

        if writer is not None:
            """ An existing writer can save memory - or even simply push the output to the file system"""
            self.writer = writer
        else:
            """ This means that we need to cache all results"""
            self.writer = DummyWriter()

        self.reduction_chain = reduction_chain
        for aug in augmentations:
            if aug.input_type() == "corpus":
                raise ValueError("Corpus based augmentation not supported yet")

    def __call__(self, corpus):
        # make call compatible with the NLPAUG package (https://github.com/makcedward/nlpaug/blob/)
        augmentations_wrapped = []
        for aug in self.augmentations:
            if isinstance(aug, BaseAugmentation):
                augmentations_wrapped.append(aug)
            else:
                meth = getattr(aug, "augment", None)
                if callable(meth):
                    augmentations_wrapped.append(NLPAugWrapper(aug))
                else:
                    raise ValueError("Provided augmentation is not implemented yet, you can add it or open an issue")
        
        for sentence in progressbar(corpus):
            docs = [sentence]
            for aug in self.augmentations:
                aug_docs = []
                for doc in docs:
                    doc_aug = aug.doc_augment(doc=doc)
                    aug_docs += doc_aug
                docs = aug_docs

                log.debug(f"{self.__class__.__name__}: DOCs={docs}")
                log.debug(f"{self.__class__.__name__}: AUG={aug},  len={len(docs)}")

            for reduction in self.reduction_chain:
                docs = reduction(docs)

            for d in docs:
                self.writer.write(d)

        return self.writer.close()


class NLPAugWrapper(SingleprocessingAugmentation):
    """
    This wraps a class of the package `nlpaug` (https://pypi.org/project/nlpaug/), to make it integrable in the
    augmentation pipeline.
    """
    def shortname(self):
        return str(self.nlp_aug_object)

    def __init__(self, nlp_aug_object):
        self.nlp_aug_object = nlp_aug_object

    def get_config(self):
        return dict(class_name=str(self), hint="This wrappes NLPAug objects")

    def augmentation_implementation(self, sentence):
        return self.nlp_aug_object.augment(sentence)

    def input_type(self):
        return "text"
