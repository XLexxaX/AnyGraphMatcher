import logging
from scipy import linalg
import numpy as np
# import torch
# import torch.nn as nn

from sklearn import preprocessing
from crosslingual.util import project_embeddings_to_lexicon_subset, create_keyed_vector

logger = logging.getLogger(__name__)


def linear_projection(word_vector_src, word_vector_tgt, lexicon):
    """
    Computes the linear projection between two word embeddings using Moore-Penrose-Pseudoinverse to solve least squares problem algebraically.
    :param word_vector_src: word embedding as KeyedVectors
    :param word_vector_tgt: word embedding as KeyedVectors
    :param lexicon: iterable of (source_word, target_word)
    :return: the projected source embedding
    """
    matrix_src, matrix_tgt = project_embeddings_to_lexicon_subset(word_vector_src, word_vector_tgt, lexicon)

    x_mpi = linalg.pinv(matrix_src)  # Moore Penrose Pseudoinverse
    w = np.dot(x_mpi, matrix_tgt)  # linear map matrix W

    source_projected = create_keyed_vector(word_vector_src, np.dot(word_vector_src.wv.vectors, w))
    return source_projected


# def sgd_projection(word_vector_src, word_vector_tgt, lexicon, epochs=3000, learning_rate=0.01):
#     matrix_src, matrix_tgt = project_embeddings_to_lexicon_subset(word_vector_src, word_vector_tgt, lexicon)
#
#     class LRModel(nn.Module):
#
#         def __init__(self, in_dim, out_dim):
#             super(LRModel, self).__init__()
#             self.linear = nn.Linear(in_dim, out_dim, bias=False)
#
#         def forward(self, x):
#             out = self.linear(x)
#             return out
#
#     model = LRModel(word_vector_src.vector_size, word_vector_tgt.vector_size)
#     criterion = nn.MSELoss()  # MSE loss corresponds to objective function
#     optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
#     for epoch in range(epochs):
#         epoch += 1
#         optimiser.zero_grad()
#         output = model.forward(torch.Tensor(matrix_src))
#         loss = criterion(output, torch.Tensor(matrix_tgt))
#         loss.backward()
#         optimiser.step()
#         if epoch % 100 == 0:
#             CONFIGURATION.log('epoch {}, loss {}'.format(epoch, loss.data[0]))
#
#     w = list(model.parameters())[0]
#     w = np.transpose(torch.tensor(w).detach().numpy())
#
#     source_projected = create_keyed_vector(word_vector_src, np.dot(word_vector_src.wv.vectors, w))
#     return source_projected


def orth_projection(word_vector_src, word_vector_tgt, lexicon, fill_value=0.01):
    matrix_src, matrix_tgt = project_embeddings_to_lexicon_subset(word_vector_src, word_vector_tgt, lexicon)

    X = matrix_src.copy()
    Z = matrix_tgt.copy()

    if X.shape[1] < Z.shape[1]:
        X = np.concatenate((X, np.full((X.shape[0], Z.shape[1] - X.shape[1]), fill_value)), axis=1)
        X = preprocessing.normalize(X, axis=1, norm='l2')
        full_src = np.concatenate((word_vector_src.wv.vectors, np.full((word_vector_src.wv.vectors.shape[0], Z.shape[1] - word_vector_src.wv.vectors.shape[1]), fill_value)), axis=1)
        full_src = preprocessing.normalize(full_src, axis=1, norm='l2')
    elif X.shape[1] > Z.shape[1]:
        Z = np.concatenate((Z, np.full((Z.shape[0], X.shape[1] - Z.shape[1]), fill_value)), axis=1)
        Z = preprocessing.normalize(Z, axis=1, norm='l2')
        full_src = word_vector_src.wv.vectors
    else:
        full_src = word_vector_src.wv.vectors

    M = np.dot(np.transpose(X), Z)
    U, S, V_T = linalg.svd(M, full_matrices=True)
    S_1s = np.diag(np.append(np.ones((S > 0).sum()), (np.zeros(min(U.shape[1], V_T.shape[0]) - (S > 0).sum()))))
    w = U @ S_1s @ V_T

    source_projected = create_keyed_vector(word_vector_src, np.dot(full_src, w))
    return source_projected


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
    logging.info("Start")

    from eval.evaldbkwik import eval_dbkwik

    #eval_dbkwik(RandomTranslationMatcher(generate_random_walks, replacement_chance=0.1),
    #            selection=set([('darkscape', 'oldschoolrunescape')]),
    #            inital_mapping_gold_standard=0.5)

