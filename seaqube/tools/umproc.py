"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns

This file is thankfully copied from https://github.com/thouska/spotpy
    (and also with pleasure to the cool project and team)
"""
import pathos.multiprocessing as mp


class ForEach(object):
    """
    ForEach is a classes for multiprocessed work based on a generater object which is given if __call__ is called
    We using the pathos multiprocessing module and the orderd map function where results are saved until results in
    the given order are caluculated. We yielding back the result so a generator object is created.
    """
    def __init__(self, process, max_cpus=None):
        self.size = mp.cpu_count()

        if max_cpus is not None and self.size > max_cpus:
            self.size = max_cpus

        self.process = process
        self.pool = mp.ProcessingPool(self.size)

    def is_idle(self):
        return False

    def terminate(self):
        pass

    def start(self):
        pass

    def setphase(self, phasename):
        self.phase = phasename

    def f(self, job):
        data = self.process(job)
        return data

    def __call__(self, jobs):
        results = self.pool.uimap(self.f, jobs)
        for i in results:
            yield i