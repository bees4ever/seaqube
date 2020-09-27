# SeaQuBe
Semantic Quality Benchmark for Word Embeddings, i.e. Natural Language Models in Python. Acronym `SeaQuBe` or `seaqube`.

<img src="https://travis-ci.org/bees4ever/SeaQuBe.svg?branch=master&amp;status=started" alt="build:started">

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


## Basic Example

*See also: examples/basic_augmentation jupyter notebook*


#### Import all Augmentation methods
````python
from seaqube.augmentation.word import Active2PassiveAugmentation, EDAAugmentation, TranslationAugmentation, EmbeddingAugmentation
from seaqube.augmentation.char import QwertyAugmentation
from seaqube.augmentation.corpus import UnigramAugmentation
from seaqube.tools.io import load_json
````
#### Prepare corpus and sample data
````python
text = 'The quick brown fox jumps over the lazy dog .'
corpus = load_json(join(dirname(__file__), "..", "examples", "sick_full_corpus.json"))
````

### Set up all augmentations:

#### A (experimental) active to passive voice transformer. Only one sentences / doc to another.
````python
a2p = Active2PassiveAugmentation()
````

#### Easy-data augmentation method implementation (random word swap, insertion, deletion and replacement with synonyms).
````python
eda = EDAAugmentation(max_length=2)
````

#### Translate text to other language and back (with Google Translater).
````python
translate = TranslationAugmentation(max_length=2)
````
#### Replace words by a similar one using another word embedding.
embed = EmbeddingAugmentation(max_length=2)
````

###### insert typos on text based on a qwerty-keyboard
````python
qwerty = QwertyAugmentation(replace_rate=0.07, max_length=2)
````

#### Based on the UDA algorithm, only the Unigram method, which replace low meaning full words with other low meaning full words. This method needs a corpus, because it need to detect low meaningfull words
````python
unigram = UnigramAugmentation(corpus=corpus, max_length=2)
````


### API - Usage
#### Every augmentation object have the same possibility
````python
# 1. augmenting a string - same syntax as NLPAUG (https://github.com/makcedward/nlpaug)
print(qwerty.augment(text))
# or
print(translate.augment(text))

# 2. augmenting a doc (token based text)
print(unigram.doc_augment(doc=corpus[0]))
# doc_augment can also handle text:
print(embed.doc_augment(text=text))

# 3. augmenting a whole corpus
print(eda(corpus[0:200]))

# 4. Active2Passive is still experimental:
a2p.doc_augment(doc=['someone', 'is', 'not', 'reading', 'the', 'email'])
````


#### We want to apply a method on a corpus, train a model and meassure the performance
````python
# tidy up RAM
del unigram, embed, translate
corpus_augmented = eda(corpus[0:200]) # augment a small subset

# save on disk:
#save_json(corpus_augmented, "augmented_sick.json")

# To use NLP models which matching to or benchmark tool set, it must implement the 'BaseModelWrapper' interface.
# We set up a class who implements the fasttext nlp model from the gensim package.
# This is only needed to get the benchmark run

from gensim.models import FastText

class FTModelStd500V5(BaseFTGensimModel):
    def define_epochs(self):
        return 100

    def define_model(self):
        return FastText(sg=1, cbow_mean=1, size=300, alpha=0.025, min_alpha=0.0001, min_n=1, max_n=5,
                        window=5, min_count=1, sample=0.001, negative=5, workers=self.cpus - 1)

    def define_training(self):
        self.model.build_vocab(sentences=self.data, update=False)
        self.model.train(sentences=self.data, total_examples=len(self.data), epochs=self.epochs)

model = FTModelStd500V5()

# train the model
# model.train_on_corpus(corpus_augmented)

# get a dumped model to store it on disk - or use it in another process
# model.get()
# dill_dumper(model.get(), "example_model.dill")
# or to save a compressed model:
# SeaQuBeCompressLoader.save_model_compressed(model.get(), "example_model_compressed.dill")
nlp = SeaQuBeCompressLoader.load_compressed_model(join(dirname(__file__), "..", "examples", "example_model_compressed.dill"), "example")

del model
````

#### Perform the Semantic Quality Analysis with Benchmark Tools
````python
from seaqube.benchmark.corpus4ir import Corpus4IRBenchmark
from seaqube.benchmark.wordanalogy import WordAnalogyBenchmark
from seaqube.benchmark.wordsimilarity import WordSimilarityBenchmark

wsb = WordSimilarityBenchmark(test_set='simlex999')
print(wsb(nlp.model))  # score=0.008905456556563954

wab = WordAnalogyBenchmark('google-analogies')
print(wab(nlp.model))  # score=0.0

c4ir = Corpus4IRBenchmark(corpus[0:200])  # need the original corpus for setting up IR
print(c4ir(nlp.model))
````


## Setup Dev Environment

#### Tools

````bash
 npm install generate-changelog -g 
 # see: https://www.npmjs.com/package/generate-changelog
````
