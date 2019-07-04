from scipy import linalg
import scipy.io as sio
import numpy as np
from cle.crosslingual.util import project_embeddings_to_lexicon_subset, create_keyed_vector


def __normr(arr):
    return arr / linalg.norm(arr, axis=1, ord=2, keepdims=True)


def __canoncorr(X, Y):
    # Based on sourceforge.net/p/octave/statistics/ci/release-1.4.0/tree/inst/canoncorr.m

    #sio.savemat('np_vector.mat', {'X': X, 'Y': Y})
    #additional constraint because otherwise line ' A = linalg.solve(Rx, U[:, :d]) ' does not work
    assert (X.shape[0] > X.shape[1] and Y.shape[0] > Y.shape[1]), \
        'Vector dimension must be greater than trainings lexicon - maybe decrease vector size.'

    k = X.shape[0]
    m = X.shape[1]
    n = Y.shape[1]
    d = min(m, n)

    assert (X.shape[0] == Y.shape[0])  # both array should have same number of rows


    X = X - X.mean(axis=0, keepdims=True)  # center X = remove mean
    Y = Y - Y.mean(axis=0, keepdims=True)  # center Y = remove mean

    Qx, Rx = linalg.qr(X, mode='economic')
    Qy, Ry = linalg.qr(Y, mode='economic')

    U, S, V = linalg.svd(Qx.T.dot(Qy),
                         full_matrices=False)  # full_matrices=False should correspind to svd(...,0)   #, lapack_driver='gesvd'
    V = V.T  # because svd returns transposed V (called Vh)

    A = linalg.solve(Rx, U[:, :d])
    B = linalg.solve(Ry, V[:, :d])

    f = np.sqrt(k - 1)
    A = np.multiply(A, f)
    B = np.multiply(B, f)

    return A, B


def cca_projection(word_vector_source, word_vector_target, lexicon, top_correlation_ratio = 0.5):
    word_vector_source.init_sims(replace=True) # TODO: yes no maybe?
    word_vector_target.init_sims(replace=True)
    # all vectors are normalised because of the above function calls and the subset is also normalised because it is a subset

    source_subset, target_subset = project_embeddings_to_lexicon_subset(word_vector_source, word_vector_target, lexicon)

    A, B = __canoncorr(target_subset, source_subset)  # maybe other way around?

    amount_A = int(np.ceil(top_correlation_ratio * A.shape[1]))
    U = (word_vector_target.vectors - word_vector_target.vectors.mean(axis=0, keepdims=True)).dot(A[:, 0:amount_A])
    U = __normr(U)
    projected_target_vectors = create_keyed_vector(word_vector_target, U)

    amount_B = int(np.ceil(top_correlation_ratio * B.shape[1]))
    V = (word_vector_source.vectors - word_vector_source.vectors.mean(axis=0, keepdims=True)).dot(B[:, 0:amount_B])
    V = __normr(V)
    projected_source_vectors = create_keyed_vector(word_vector_source, V)

    return projected_source_vectors, projected_target_vectors