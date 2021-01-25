# pip install transformers==2.8.0
import string
from copy import deepcopy
from functools import reduce
from os.path import join, basename
import random
from typing import List
import logging
import numpy
import torch
from tokenizers import ByteLevelBPETokenizer
import os
from transformers import AutoModelWithLMHead, AutoTokenizer
import json
from sklearn.model_selection import train_test_split

from seaqube.nlp.roberta.roberta_training import RoBERTaSmall

from seaqube.nlp.tools import sentenceize_corpus
from seaqube.nlp.types import SeaQuBeWordEmbeddingsModel, SeaQuBeNLPModel2WV

logger = logging.getLogger("SEABERTA")

class SeaBERTaWV:
    def __init__(self, vocabs: dict, matrix, tokenizer: AutoTokenizer):
        self.word2index = vocabs
        self.vocabs: list = list(vocabs.keys())
        self.matrix = matrix
        self.index2word = self.vocabs
        self.vectors = matrix
        self.tokenizer: AutoTokenizer = tokenizer

    def __getitem__(self, word: str):
        word = 'Ġ' + word.strip()
        try:
            return self.matrix[self.word2index[word]]
        except KeyError:
            token_ids = self.tokenizer.tokenize(word.strip())
            return numpy.sum([self.matrix[self.word2index[token]].numpy() for token in token_ids], axis=0)


class SeaBERTa:
    """ A RoBERTa wrapper to easily train, save and load a RoBERTa model"""
    def __init__(self, main_path: str, train_params: dict):
        self.corpus = None
        self.main_path = main_path
        self.train_params = train_params
        self.__wv = None
        self.dir_cache = f"{self.main_path}/cache"
        os.makedirs(self.dir_cache, exist_ok=True)

    def to_txt(self, corpus, filename):
        path = join(self.main_path, filename)
        if not os.path.exists(path):
            corpus = list(map(lambda x: " " + " ".join(x), corpus))
            with open(path, "w", encoding="utf-8") as f:
                f.writelines("\n".join(corpus))

    @property
    def dataset_name(self):
        return basename(self.main_path)

    def prepare_sets(self):
        corpus = deepcopy(self.corpus)
        train, eval_set = train_test_split(corpus, shuffle=True)

        self.to_txt(corpus, f"{self.dataset_name}.txt")
        self.to_txt(train, f"{self.dataset_name}_train.txt")
        self.to_txt(eval_set, f"{self.dataset_name}_eval.txt")

        # calc vocab size

        return len(set(reduce(lambda a, b: a + b, corpus)))

    def train(self, corpus):
        self.corpus = corpus

        raw_vocab_path = f"{self.main_path}/{self.dataset_name}.txt"
        vocab_size = self.prepare_sets() + 1000

        config = {
            "architectures": [
                "RobertaForMaskedLM"
            ],
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-05,
            "max_position_embeddings": 514,
            "model_type": "roberta",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 1,
            "vocab_size": vocab_size
        }

        with open(f"{self.main_path}/config.json", 'w', encoding="utf-8") as fp:
            json.dump(config, fp)

        print(raw_vocab_path, vocab_size)

        # Initialize a tokenizer
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=raw_vocab_path,
                        vocab_size=vocab_size,
                        min_frequency=2,
                        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

        try:
            tokenizer.save_model(self.main_path)
        except Exception:
            tokenizer.save(self.main_path)

        tokenizer_config = {"max_len": 512}

        with open(f"{self.main_path}/tokenizer_config.json", 'w', encoding="utf-8") as fp:
            json.dump(tokenizer_config, fp)

        # model paths
        self.train_params["output_dir"] = f"{self.main_path}/output"
        self.train_params["model_type"] = "roberta"
        self.train_params["config_name"] = self.main_path
        self.train_params["tokenizer_name"] = self.main_path
        self.train_params["train_data_file"] = f"{self.main_path}/{self.dataset_name}_train.txt"
        self.train_params["eval_data_file"] = f"{self.main_path}/{self.dataset_name}_eval.txt"
        self.train_params["cache_dir"] = self.dir_cache

        roberta = RoBERTaSmall()
        roberta.train(self.train_params, config)

        # train is done load all the data!!
        self.load_trained_model()

    def load_trained_model(self):
        main_output_dir = f"{self.main_path}/output"
        self.model = AutoModelWithLMHead.from_pretrained(main_output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(main_output_dir)
        weights = self.model.roberta.embeddings.word_embeddings.state_dict()["weight"]
        # vocab file
        with open(f"{self.main_path}/vocab.json", "r", encoding="utf-8") as f:
            self.vocabs: dict = json.load(f)

        matrix_raw = []
        vocabs_collected = []
        for vocab, line in self.vocabs.items():
            if vocab[0] == 'Ġ':
                matrix_raw.append(weights[line].numpy())
                vocabs_collected.append(vocab.replace('Ġ', ''))

        self.__wv = SeaQuBeNLPModel2WV(vocabs_collected, numpy.array(matrix_raw))

    def context_embedding(self, words, position):
        """
        This method calculates the word embeddings based on the context, i.e. the last hidden state of RoBERTa.
        To do so, the model and tokenizer needs to be loaded
        """

        words = [word.lower() for word in words]
        text = ' ' + sentenceize_corpus([words])[0]

        logger.debug(f"[context_embedding] words: {words}")
        logger.debug(f"[context_embedding] text: {text}")
        snippets = self.tokenizer.encode(text)[1:-1]

        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        outputs = self.model(input_ids)
        last_hidden_states = outputs[0]

        index_2_word = {v: k for k, v in self.vocabs.items()}

        splitted = [index_2_word[index] for index in snippets]
        logger.debug(f"[context_embedding] splitted: {splitted}")
        logger.debug(f"[context_embedding] snippets: {snippets}")

        pos_in_token = 0

        summarized_indices = [[] for _ in range(len(words))]

        tmp_orig_tokens = deepcopy(words)
        for j, word_part in enumerate(splitted):
            word_part = word_part.replace('Ġ', '')

            #print("debug 1", word_part, j, pos_in_token, tmp_orig_tokens[pos_in_token])
            if word_part not in tmp_orig_tokens[pos_in_token]:
                pos_in_token += 1

            if word_part in tmp_orig_tokens[pos_in_token]:
                #print("IN?")
                summarized_indices[pos_in_token].append(j + 1)
                tmp_orig_tokens[pos_in_token] = tmp_orig_tokens[pos_in_token].replace(word_part, '', 1)
                #print("debug 2", tmp_orig_tokens[pos_in_token])

        new_sorted_wes = []
        for _list in summarized_indices:
            new_sorted_wes.append(numpy.mean(last_hidden_states[0][_list].detach().numpy(), axis=0))

        logger.debug(f"[context_embedding] new_sorted_wes: {new_sorted_wes}")
        return numpy.array(new_sorted_wes)[position]


    @property
    def wv(self):
        return self.__wv


class SeaQuBeWordEmbeddingsModelSeaBERTa(SeaQuBeWordEmbeddingsModel):
    def __init__(self, seaberta: SeaBERTa):
        self.seaberta = seaberta

    def vocabs(self) -> List[str]:
        return self.seaberta.wv.vocabs

    @property
    def wv(self):
        return self.seaberta.wv

    def word_vector(self, word):
        return self.seaberta.wv[word]

    def matrix(self):
        return self.seaberta.wv.matrix

    def context_embedding(self, words, position):
        return self.seaberta.context_embedding(words, position)