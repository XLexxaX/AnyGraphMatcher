import numpy as np
import scipy.io as sio
from wordembedding.cooccurmat import get_occurrence, create_sparse_matrix
from itertools import chain
from collections import defaultdict
from scipy import sparse
from gensim.models.word2vec import LineSentence

# class my_iter:
#     def __init__(self, iter_one, iter_two):
#         self.iter_one = iter_one
#         self.iter_two = iter_two
#
#     def __iter__(self):
#         for i in chain(self.iter_one, self.iter_two):
#             yield i

def translation_invariant_lsa(sentences_one, sentences_two, lexicon, window=5, min_count=None, n_most_frequent=None, size=100):
    # occurrence_one, vocab_one = get_occurrence(sentences_one, None, window, min_count, n_most_frequent)
    # occurrence_two, vocab_two = get_occurrence(sentences_two, None, window, min_count, n_most_frequent)
    #
    # occurrence_one = occurrence_one.todense()
    # occurrence_two = occurrence_two.todense()
    #
    # #https://stackoverflow.com/questions/6844998/is-there-an-efficient-way-of-concatenating-scipy-sparse-matrices/10557162
    #
    # right_upper = np.zeros((occurrence_one.shape[0], occurrence_two.shape[1]))
    # left_down = np.zeros((occurrence_two.shape[0], occurrence_one.shape[1]))
    # X = np.block([
    #     [occurrence_one, right_upper],
    #     [left_down,      occurrence_two]
    # ])
    #
    # for src_word, target_word in lexicon:

    with open('tmp.txt', 'w', encoding='utf-8') as f:
        for sentence in chain(sentences_one, sentences_two):
            f.write(' '.join(sentence) + '\n')

    occurrence, vocab = get_occurrence(LineSentence('tmp.txt'), None, window, min_count, n_most_frequent, dtype=np.float64) #or float32

    d1 = defaultdict(int)  # coocurrance[(row,col)] = data
    for src_word, target_word in lexicon.items():
        src_id, target_id = vocab.get_word_ids([src_word, target_word])
        d1[src_id, target_id] = 1
        d1[target_id, src_id] = 1
    d1_matrix = create_sparse_matrix(d1, np.uint32, occurrence.shape)

    vector_matrix = __dxd_svd(occurrence, d1_matrix, d1_matrix, size=size)



    #sio.savemat('multilingual.mat', {'X': occurrence, 'D1': d1_matrix, 'D2': d1_matrix})
    #dxd_svd(occurrence, d1_matrix, d1_matrix)

    # test = np.concatenate([word_vector_src.vectors, word_vector_tgt.vectors])
    #
    # dxd_svd()
    #
    # CONFIGURATION.log(word_vector_src.vectors)

# TODO: check that return type is float32 or float64


def __dxd_u(X, P1, P2, u, m, n):
    #sio.savemat('dxd_u.mat', {'X': X, 'P1': P1, 'P2': P2, 'u': u, 'm': m, 'n': n})
    u1 = u[:n]
    u2 = u[n:m + n]
    uu1 = P2 * (X.T * (P1.T * u2))
    uu2 = P1 * (X * (P2.T * u1))
    u = np.concatenate([uu1, uu2])
    return u


def __dxd_svd(X, D1, D2, size=100, lam=1):
    #sio.savemat('multilingual.mat', {'X': X, 'D1': D1, 'D2': D2})
    m, n = X.shape
    P1 = sparse.eye(m) + lam * D1
    P2 = sparse.eye(n) + lam * D2

    lo = sparse.linalg.LinearOperator(shape=(m+n, m+n), matvec=lambda u : __dxd_u(X, P1, P2, u, m, n), dtype=np.float64)
    vals, vecs = sparse.linalg.eigsh(lo, k=2*size, which='LM')
    Ss = np.diagflat(vals[:size])
    Vs = np.sqrt(2) * vecs[:n, :size]
    Us = np.sqrt(2) * vecs[n:n+m, :size]

    #CONFIGURATION.log(Us)
    CONFIGURATION.log(Ss)
    CONFIGURATION.log(Vs)
    CONFIGURATION.log(Us)

    word_vector = Us.dot(Ss)
    #CONFIGURATION.log(word_vector.shape)
    #CONFIGURATION.log(word_vector)
    #CONFIGURATION.log(Q)

    return word_vector


if __name__ == '__main__':
    np.set_printoptions(linewidth=np.nan, suppress=True)

    common_texts_src = [
        ['human', 'interface', 'computer'],
        ['survey', 'user', 'computer', 'system', 'response', 'time']
    ]

    common_texts_dst = [
        ['ordinateur ', 'dinterface', 'humain'],
        ['temps', 'de', 'reponse', 'du', 'systeme', 'informatique', 'utilisateur', 'denquete' ],
    ]

    # translation_invariant_lsa(common_texts_src, common_texts_dst, {
    #     'human':'humain',
    #     'time': 'temps',
    #     'system':'systeme'
    # })

    #test = sio.loadmat('multilingual.mat')
    #CONFIGURATION.log(__dxd_svd(test['X'], test['D1'], test['D2'], size=10))

    #test = sio.loadmat('dxd_u.mat')
    #dxd_u(test['X'], test['P1'], test['P2'], test['u'], test['m'], test['n'])

    # f = np.dtype("f")
    # bf = np.dtype("F")
    # d = np.dtype("d")
    # bd = np.dtype("D")
    # CONFIGURATION.log("bla")


