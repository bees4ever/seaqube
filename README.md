# SeaQuBe
Semantic Quality Benchmark for Word Embeddings, i.e. Natural Language Models in Python. Acronym `SeaQuBe` or `seaqube`.


# Introduction

The idea of this package is to provide text data augmentation strategies to boost semantic word embedding quality. Some text Augmentation strategies are already available and fit into the usage of this package: https://github.com/makcedward/nlpaug.

However, this package provide also implementation of 


## Installation

`SeaQuBe` can be installed from PyPip using: `pip install seaqube`. 

External lib with:

# todo: nlp.model.wv.__dict__['index2word'] = nlp.model.vocabs()
# nlp.model.wv.__dict__['vectors'] = nlp.model.matrix()


and with:

python
>>> import nltk; nltk.download('wordnet')


# Word Embeddings Quality

Standart Datset provided from: https://github.com/vecto-ai


# Usage of self generated models with NLP Loader:
--> This makes it easier
nlp = SeaQuBeNLPLoader.load_model_from_tin_can(model.get(), 'small_model')
nlp("Hi")