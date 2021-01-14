"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns

NLP IO Toolkit (JSON and Compressed data)
"""
import json
import dill
from seaqube.tools.types import Writeable


class DummyWriter(Writeable):
    def __init__(self):
        self.cache = []

    def write(self, data):
        self.cache.append(data)

    def close(self):
        return self.cache


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path: str):
    with open(path, "w") as f:
        return json.dump(data, f)


def dill_dumper(obj, path):
    with open(path, "wb") as f:
        dill.dump(obj, f)