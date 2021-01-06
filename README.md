<p align="center">
    <br>
    <img width="200px" src="https://github.com/bees4ever/SeaQuBe/raw/master/logo/seaqube_logo_v1.png"/>
    <br>
<p>

# SeaQuBe

Semantic Quality Benchmark for Word Embeddings, i.e. Natural Language Models in Python. Acronym `SeaQuBe` or `seaqube`.

This python framework provides several text augmentation implementations and word embedding quality evaluation methods. It is designed to fit in your machine learning pipeline. The `BaseAugmentation` class provides the same api as the python package [nlpaug](https://github.com/makcedward/nlpaug/), so that this packages can used together smoothly. However `BaseAugmentation` provides also other methods. Detailed examples see beneath.

`SeaQuBe` provides also a toolkit to wrap a trained nlp model to a nice interactive tool.

<img src="https://travis-ci.org/bees4ever/SeaQuBe.svg?branch=master&amp;status=started" alt="build:started">

## Features

*  Text Data Augmentation
*  Chaining and Reducing of Text Data Augmentations
*  Word Embedding Quality Methods
*  Interactive NLM Model Wrapper

## Demo
*   [Augmentation in three lines](https://github.com/bees4ever/SeaQuBe#quick-demo)
*   [Example of Basic Text Augmentation](https://github.com/bees4ever/SeaQuBe/blob/master/examples/basic_augmentation.ipynb)
*   [Example of Text Augmentation Chaining](https://github.com/bees4ever/SeaQuBe/blob/master/examples/chained_augmentation.ipynb)
*   [Example of Word Embedding Evaluation](https://github.com/bees4ever/SeaQuBe/blob/master/examples/word_embedding_evaluation.ipynb)
*   [Example of Interactive NLP](https://github.com/bees4ever/SeaQuBe/blob/master/examples/nlp.ipynb)

## Augmentation
| Level  | Augmenter  | Description |
|:---:|:---:|:---:|
| Character | QwertyAugmentation | Simulate keyboard distance error |
| Corpus | UnigramAugmentation | Replace ubiquitous words with other ubiquitous words |
| Word | Active2PassiveAugmentation | Change surface of document using an simple active-to-passive transformer |
| Word | EDAAugmentation | Augment document using the [EDA](https://github.com/jasonwei20/eda_nlp) algorithm |
| Word | EmbeddingAugmentation | Replace similar word using [WordNet](https://wordnet.princeton.edu/) |
| Word | TranslationAugmentation | Change surface of document using translation and back-translation (with [GoogleTranslate](https://translate.google.com/))|

## Augmentation Chainer
The streaming feature of augmentation is implemented in the ``AugmentationStreamer`` class. One `Reduceing` class exist, more can implemented
extending the ``BaseReduction`` class.  

| Action  | Class  | Description |
|:---:|:---:|:---:|
|Streaming|AugmentationStreamer| Run augmentation for each document through all chained augmentations.  |
|Reducing| UniqueCorpusReduction | Getting a list of documents, only unique documents are returned.  

## Word Embedding Evaluation
| Method  | Description |
|:---:|:---:|
|WordAnalogyBenchmark|This method benchmark how go relations of the type: `a is to b as c is to d` can be solved correctly.|
|WordSimilarityBenchmark|This methods compares the similarity of a word pair, calculated by a model with a human estimated similarity score.|
|WordOutliersBenchmark|This method benchmark how good a outlier of a group of words can be detected.|
|SemanticWordnetBenchmark|Based on the WordNet graph, the goodnes of the semantic / similarity of a nlp model is benchmarked.|

## Installation

`SeaQuBe` can be installed from PyPip using: `pip install seaqube` or run in the main directory: `python setup.py install`.

### External Dependencies

Some external dependencies are not installed automatically, but `seaqube` or `nltk` might throw errors with an instruction what to do.
For example ``seqube`` might ask you to run:

````bash 
python -c "from seaqube import download;download('vec4ir')"
````

## Quick Demo
````python
from seaqube.augmentation.word import Active2PassiveAugmentation, EDAAugmentation, TranslationAugmentation, EmbeddingAugmentation
translate = TranslationAugmentation(max_length=2)
translate.doc_augment(['This', 'is', 'a', 'tokenized', 'corpus'])
````

## Setup Dev Environment
_TODO_