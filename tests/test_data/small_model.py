'''
Copyright (c) 2020 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
'''

from gensim.models import FastText
from seaqube.nlp.seaqube_model import BaseModelWrapper
from seaqube.nlp.tools import gensim_we_model_to_custom_we_model
from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.models.callbacks import CallbackAny2Vec
from progressbar import ProgressBar


class GensimEpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self, epochs):
        self.epoch = 0
        self.bar = ProgressBar(max_value=epochs)

    def on_train_begin(self, model: BaseWordEmbeddingsModel):
        pass

    def on_epoch_begin(self, model):
        pass

    def on_epoch_end(self, model):
        self.epoch += 1
        self.bar.update(self.epoch)


class BaseFTGensimModel(BaseModelWrapper):
    def get_config(self):
        return dict(sg=self.model.sg, cbow_mean=self.model.cbow_mean, size=self.model.vector_size,
                    alpha=self.model.alpha, min_alpha=self.model.min_alpha, min_n=self.model.wv.min_n,
                    max_n=self.model.wv.max_n, window=self.model.window, min_count=self.model.vocabulary.min_count,
                    sample=self.model.vocabulary.sample, negative=self.model.negative, workers=self.model.workers,
                    class_name=str(self))

    def _wrap_nlp_model(self, model):
        return gensim_we_model_to_custom_we_model(model)


class SmallModel(BaseFTGensimModel):
    def define_epochs(self):
        return 1000

    def define_model(self):
        return FastText(sg=1, cbow_mean=1, size=200, alpha=0.025, min_alpha=0.0001, min_n=1, max_n=6,
                        window=5, min_count=1, sample=0.001, negative=5, workers=self.cpus - 1,
                        callbacks=[GensimEpochLogger(self.epochs)])

    def define_training(self):
        self.model.build_vocab(sentences=self.data, update=False)
        self.model.train(sentences=self.data, total_examples=len(self.data), epochs=self.epochs)
