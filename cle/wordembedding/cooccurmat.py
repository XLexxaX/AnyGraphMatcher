import logging
import numpy as np
from collections import defaultdict
from scipy.sparse import csc_matrix
from wordembedding.worddictionary import WordDictionary


def create_sparse_matrix(x_y_pos_to_value_dict, dtype, shape=None):
    # create csc matrix
    # keys and vaues correspond (some order) see
    # https://docs.python.org/2/library/stdtypes.html#dict.items
    # https://docs.python.org/3/library/stdtypes.html#dictionary-view-objects
    dict_len = len(x_y_pos_to_value_dict)
    data = np.fromiter(x_y_pos_to_value_dict.values(), dtype=dtype, count=dict_len)
    I = np.fromiter((i for i, _ in x_y_pos_to_value_dict.keys()), dtype=np.uint32, count=dict_len)
    J = np.fromiter((j for _, j in x_y_pos_to_value_dict.keys()), dtype=np.uint32, count=dict_len)
    return csc_matrix((data, (I, J)), shape)


def get_occurrence(sentence_generator, vocab=None, window=5, min_count=None, n_most_frequent=None, dtype=np.uint32):
    if vocab is None:
        vocab = WordDictionary(sentence_generator)
        vocab.filter_extremes(min_count, n_most_frequent)

    coocurrance = defaultdict(int)  # coocurrance[(row,col)] = data
    for i, sentence in enumerate(sentence_generator):
        if i % 10000 == 0:
            logging.info("processing sentence #%i", i)

        # return_missing = remove all out of vocabulary (OOV) words
        word_ids = vocab.get_word_ids(sentence, return_missing=False)
        if len(word_ids) <= 1:
            continue

        for pos, word_id in enumerate(word_ids):
            # reduced_window: see https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py#L186
            start = max(0, pos - window)
            for pos2, word2_id in enumerate(word_ids[start:(pos + window + 1)], start):
                if pos2 == pos:
                    continue  # don't train on the `word` itself
                coocurrance[(word_id, word2_id)] += 1

    return create_sparse_matrix(coocurrance, dtype), vocab