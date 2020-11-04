import random
from itertools import product

from seaqube.augmentation.word.active2passive import Active2PassiveAugmentation
from seaqube.tools.math import lazy_sample

if __name__ == "__main__":
    # r = random.Random()
    # r.seed(123)
    # P = product(range(10), repeat=100)
    # L1 = lazy_sample(P, 10 ** 100, 10, r)
    # print(list(L1))


    a2p = Active2PassiveAugmentation()
    import logging

    logging.basicConfig(level=logging.DEBUG)
    def active2passive(text):
        return a2p.doc_augment(text=text)[0]

    #print(active2passive("I was waiting for small Dina"))
    print(active2passive("Simson cares little Simon"))

    def bla():
        yield 1
        raise StopIteration()