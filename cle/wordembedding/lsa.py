from cle.wordembedding.cooccurmat import get_occurrence
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
from scipy.sparse import linalg
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Vocab
from operator import itemgetter
from gensim.models.word2vec import LineSentence


def __multiply_by_rows(matrix, row_coefs):
    normalizer = dok_matrix((len(row_coefs), len(row_coefs)))
    normalizer.setdiag(row_coefs)
    return normalizer.tocsr().dot(matrix)


def __multiply_by_columns(matrix, col_coefs):
    normalizer = dok_matrix((len(col_coefs), len(col_coefs)))
    normalizer.setdiag(col_coefs)
    return matrix.dot(normalizer.tocsr())


def __to_pmi(cooccurrence, log=True, k_shift=None, positive_values=True):
    #https://github.com/piskvorky/word_embeddings/blob/master/run_embed.py#L169
    #https://bitbucket.org/omerlevy/hyperwords/src/688addd64ca2ce8b4772be317f0b980f7716f4d6/hyperwords/counts2pmi.py?at=default&fileviewer=file-view-default

    #from Levy & Goldberg
    #https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf
    sum_w = np.array(cooccurrence.sum(axis=1))[:, 0]
    sum_c = np.array(cooccurrence.sum(axis=0))[0, :]
    sum_total = sum_c.sum() # TODO: check if sum_c is correct and not sum_w
    sum_w = np.reciprocal(sum_w)
    sum_c = np.reciprocal(sum_c)
    pmi = csr_matrix(cooccurrence)
    pmi = __multiply_by_rows(pmi, sum_w) #(w, c) / #w
    pmi = __multiply_by_columns(pmi, sum_c)  #(w, c) / (#w * #c)
    pmi = pmi * sum_total #(w, c) * D / (#w * #c)

    if log:
        pmi.data = np.log(pmi.data) # PMI = log(#(w, c) * D / (#w * #c))

    if k_shift is not None:
        pmi.data = pmi.data - np.log(k_shift) # shift: SPPMIk(w, c) = max(PMI(w, c) − log k, 0)

    if positive_values:
        pmi = np.ceil(pmi) # PPMI(w, c) = max(PMI (w, c), 0)

    return pmi


def _compute_svd(coocurrance_matrix, size, s_exponent=None):
    U, s, Vh = linalg.svds(coocurrance_matrix, k=size)
    if s_exponent is not None:
        word_vectors = U * (s ** s_exponent)
    else:
        word_vectors = U * s
    return word_vectors


def __create_keyed_vector(matrix, orig_vocab):
    vocab = dict()
    index_to_word = []
    for word, word_id in sorted(orig_vocab.token2id.items(), key=itemgetter(1)):
        index_to_word.append(word)
        vocab[word] = Vocab(index=word_id, count=orig_vocab.word_freq[word_id])
    vector_size = matrix.shape[1]

    keyed_vector = KeyedVectors(vector_size)
    keyed_vector.vector_size = vector_size
    keyed_vector.vocab = vocab
    keyed_vector.index2word = index_to_word
    keyed_vector.vectors = matrix
    assert (len(vocab), vector_size) == keyed_vector.vectors.shape
    return keyed_vector


def lsa_embedding_from_sentences(sentence_generator, vocab=None, size=100, window=5, min_count=None, n_most_frequent=None, pmi_log=True, pmi_k_shift=None, pmi_positive_values=True, svd_s_exponent=None):
    with open('tmp.txt', 'w', encoding='utf-8') as f:
        for sentence in sentence_generator:
            f.write(' '.join(sentence) + '\n')

    occurrence_matrix, vocab = get_occurrence(LineSentence('tmp.txt'), vocab, window, min_count, n_most_frequent)
    pmi = __to_pmi(occurrence_matrix, log=pmi_log, k_shift=pmi_k_shift, positive_values=pmi_positive_values)
    word_vectors = _compute_svd(pmi, size, s_exponent=svd_s_exponent)
    keyed_vectors = __create_keyed_vector(word_vectors, vocab)
    return keyed_vectors


def lsa_embedding_from_kg(sentence_generator, **kwargs):
    return lambda kg: lsa_embedding_from_sentences(sentence_generator(kg), kwargs)