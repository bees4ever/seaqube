"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

import itertools

###### build QWERTY LOOKUP TABLE
def get_keys_around(key):
    lines = 'azertyuiop', 'qsdfghjklm', 'wxcvbn'
    line_index, index = [(i, l.find(key)) for i, l in enumerate(lines) if key in l][0]
    lines = lines[line_index - 1: line_index + 2] if line_index else lines[0: 2]
    return [
        line[index + i] for line in lines for i in [-1, 0, 1]
        if len(line) > index + i and line[index + i] != key and index + i >= 0]


from pprint import pprint


lookup_table_old = {
    '~': ['`', '1', '!', 'q'],
    '`': ['~', '1', '!', 'q'],
    '1': [],
    '!': []
}


qwerty = [
    [
        ['~', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', None], # caps off
        ['`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', None]  # caps on
    ],
    [

        [None, 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\'],
        [None, 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '{', '}', '|'],
    ],
    [

        [None, 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'", None, None],
        [None, 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ':', '"', None, None],
    ],
    [
        [None, None, 'z', 'x', 'c', 'v', 'b', 'n', 'm', '<', '>', '?', None, None],
        [None, None, 'Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/', None, None],
    ]
]

def _index_bounder(list_):
    print(list_)
    return filter(lambda x: x >= 0 and x < 13, list_)


if __name__ == "__main__":
    import string
    for key in string.ascii_lowercase:
        #print(key, get_keys_around(key))
        # string.punctuation + string.digits + string.ascii_lowercase
        pass

    #build_lookup_table()


    lookup_table = {}

    for line in range(4):
        for ul in range(2):
            for pos, char in enumerate(qwerty[line][ul]):
                chars = set()
                chars_cased = set()

                if char is not None:
                    ulpair = (ul + 1) % 2

                    for lined in [line, line-1, line+1]:
                        for posed in [pos-1, pos+1, pos]:
                            if 0 <= lined < 4 and 0 <= posed < 13:
                                if qwerty[lined][ul][posed] is not None and qwerty[lined][ul][posed] != char:
                                    chars.add(qwerty[lined][ul][posed])
                                if qwerty[lined][ulpair][posed] is not None:
                                    chars_cased.add(qwerty[lined][ulpair][posed])



                    # _index_bounder([line, line - 1, line + 1
                    #
                    #                for coordiante in _index_bounder():
                    # chars.add(qwerty[line][ul][coordiante])

                    #for coordiante in _index_bounder([pos-1, pos+1, pos]):
                    #    chars.add(qwerty[line][ulpair][coordiante])

                    lookup_table[char] = {'neighbour': list(chars), 'capsed': list(chars_cased)}

    print(lookup_table)