#!/usr/bin/env python
"""
Learns context2vec's parametric model
"""

import pickle
import time
from chainer import cuda
import chainer.links as L
import chainer.optimizers as O
from chainer.optimizer_hooks import GradientClipping
import numpy as np

from seaqube.nlp.context2vec.common.context_models import BiLstmContext
from seaqube.nlp.context2vec.common.defs import NEGATIVE_SAMPLING_NUM, IN_TO_OUT_UNITS_RATIO
from seaqube.nlp.context2vec.train.corpus_by_sent_length import read_in_corpus
from seaqube.nlp.context2vec.train.sentence_reader import SentenceReaderDict


class C2VWV:
    def __init__(self, word2index: dict, vocabs: list, matrix):
        self.matrix = matrix
        self.word2index = word2index
        self.vectors = matrix
        self.vocabs = vocabs

    def __getitem__(self, word):
        return self.matrix[self.vocabs.index(word)]


class Context2Vec:
    def __init__(self):
        self.backend_model = None
        self.target_word_units = None
        self.reader = None
        self.__training_possible = True
        self.__wv = None
        self.trainings_params = dict()


    def __run(self, epoch, optimizer):
        for epoch in range(epoch):
            begin_time = time.time()
            cur_at = begin_time
            word_count = 0
            STATUS_INTERVAL = 1000000
            next_count = STATUS_INTERVAL
            accum_loss = 0.0
            last_accum_loss = 0.0
            last_word_count = 0
            print('epoch: {0}'.format(epoch))

            self.reader.open()
            for sent in self.reader.next_batch():

                self.backend_model.zerograds()
                loss = self.backend_model(sent)
                accum_loss += loss.data
                loss.backward()
                del loss
                optimizer.update()

                word_count += len(sent) * len(sent[0])  # all sents in a batch are the same length
                accum_mean_loss = float(accum_loss) / word_count if accum_loss > 0.0 else 0.0

                if word_count >= next_count:
                    now = time.time()
                    duration = now - cur_at
                    throuput = float((word_count - last_word_count)) / (now - cur_at)
                    cur_mean_loss = (float(accum_loss) - last_accum_loss) / (word_count - last_word_count)
                    print('{} words, {:.2f} sec, {:.2f} words/sec, {:.4f} accum_loss/word, {:.4f} cur_loss/word'.format(
                        word_count, duration, throuput, accum_mean_loss, cur_mean_loss))
                    next_count += STATUS_INTERVAL
                    cur_at = now
                    last_accum_loss = float(accum_loss)
                    last_word_count = word_count

            print('accum words per epoch', word_count, 'accum_loss', accum_loss, 'accum_loss/word', accum_mean_loss)

    def train(self, sentences, trimfreq=0, ns_power=0.75, dropout=0.0, cgfile=None, gpu=-1, unit=300, batchsize=100, epoch=10, deep=True, alpha=0.001, grad_clip=None):
        if not self.__training_possible:
            raise ValueError("This model was loaded, a training afterwards is not supported, yet")

        print('GPU: {}'.format(gpu))
        print('# unit: {}'.format(unit))
        print('Minibatch-size: {}'.format(batchsize))
        print('# epoch: {}'.format(epoch))
        print('Deep: {}'.format(deep))
        print('Dropout: {}'.format(dropout))
        print('Trimfreq: {}'.format(trimfreq))
        print('NS Power: {}'.format(ns_power))
        print('Alpha: {}'.format(alpha))
        print('Grad clip: {}'.format(grad_clip))
        print('')

        context_word_units = unit
        lstm_hidden_units = IN_TO_OUT_UNITS_RATIO * unit
        self.target_word_units = IN_TO_OUT_UNITS_RATIO * unit

        if gpu >= 0:
            cuda.check_cuda_available()
            cuda.get_device(gpu).use()
        xp = cuda.cupy if gpu >= 0 else np

        prepared_corpus = read_in_corpus(sentences)

        self.reader = SentenceReaderDict(prepared_corpus, trimfreq, batchsize)
        print('n_vocab: %d' % (len(self.reader.word2index) - 3))  # excluding the three special tokens
        print('corpus size: %d' % (self.reader.total_words))

        cs = [self.reader.trimmed_word2count[w] for w in range(len(self.reader.trimmed_word2count))]
        loss_func = L.NegativeSampling(self.target_word_units, cs, NEGATIVE_SAMPLING_NUM, ns_power)

        #args = parse_arguments()
        self.backend_model = BiLstmContext(deep, gpu, self.reader.word2index, context_word_units, lstm_hidden_units, self.target_word_units, loss_func, True, dropout)

        optimizer = O.Adam(alpha=alpha)
        optimizer.setup(self.backend_model)

        if grad_clip:
            optimizer.add_hook(GradientClipping(grad_clip))

        self.trainings_params['deep'] = deep
        self.trainings_params['unit'] = unit
        self.trainings_params['dropout'] = dropout
        self.trainings_params['trimfreq'] = trimfreq
        self.trainings_params['ns_power'] = ns_power
        self.trainings_params['cgfile'] = cgfile
        self.trainings_params['gpu'] = gpu
        self.trainings_params['batchsize'] = batchsize
        self.trainings_params['epoch'] = epoch
        self.trainings_params['alpha'] = alpha
        self.trainings_params['grad_clip'] = grad_clip

        self.__run(epoch, optimizer)

    @property
    def wv(self):
        if self.__wv is None:
            return C2VWV(self.reader.word2index, list(self.reader.word2index.keys()), self.backend_model.loss_func.W.data)
        else:
            return self.__wv

    def __get_bundle(self):
        return {
            'matrix': self.backend_model.loss_func.W.data,
            'word_units': self.target_word_units,
            'index2word': self.reader.index2word,
            'word2index': self.reader.word2index,
            'backend_model': self.backend_model,
            'backend_model_params': self.trainings_params,
            'wv': self.wv
        }


    def save(self, path):
        context2vec_bundle = self.__get_bundle()

        with open(path, 'wb') as file:
            pickle.dump(context2vec_bundle, file)

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            context2vec_bundle = pickle.load(file)

        c2v = Context2Vec()
        c2v.__wv = context2vec_bundle['wv']
        c2v.__training_possible = False

        if not isinstance(context2vec_bundle['backend_model'], BiLstmContext):
            raise ValueError("Other saved models then BiLstmContext are not supported yet")

        c2v.backend_model = context2vec_bundle['backend_model']

        c2v.target_word_units = context2vec_bundle['word_units']

        return c2v