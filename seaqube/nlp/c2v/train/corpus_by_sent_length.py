'''
Converts a single large corpus file into a directory, in which for every sentence length k there is a separate file containing all sentences of that length. 
'''

from collections import Counter


def get_file(sub_files, corpus_dir, num_filename):
    if num_filename not in sub_files:
        full_file_name = corpus_dir + '/' + num_filename
        sub_files[num_filename] = open(full_file_name, 'w')        
    return sub_files[num_filename]



def read_in_corpus(corpus: list):
    max_sent_len = 128  # TODO parameter of method can replace it

    sent_counts = Counter()
    word_counts = Counter()


    batches = {} # wordnum is mapped to doc

    for words in corpus:
        wordnum = len(words)
        if 1 < wordnum <= max_sent_len:
            if wordnum not in batches.keys():
                batches[wordnum] = []

            batches[wordnum].append(words)
            sent_counts[wordnum] += 1
            for word in words:
                word_counts[word] += 1

    print('total sents read: {}\n'.format(sum(sent_counts.values())))
    print('total words read: {}\n'.format(sum(word_counts.values())))


    return {'batches': batches, 'sent_counts': sent_counts, 'word_counts': word_counts}
