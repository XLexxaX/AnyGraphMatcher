import numpy as np
from gensim.models import KeyedVectors


def project_embeddings_to_lexicon_subset(word_vector_source, word_vector_target, lexicon):
    source_subset_vectors = []
    target_subset_vectors = []
    for lang_source_word, lang_target_word in lexicon:
        if lang_source_word not in word_vector_source or lang_target_word not in word_vector_target:
            continue
        source_subset_vectors.append(word_vector_source[lang_source_word])
        target_subset_vectors.append(word_vector_target[lang_target_word])
    return np.array(source_subset_vectors), np.array(target_subset_vectors)


def create_keyed_vector(old_keyed_vector, new_matrix):
    vector_size = new_matrix.shape[1]
    keyed_vector = KeyedVectors(vector_size)
    keyed_vector.vector_size = vector_size
    keyed_vector.vocab = old_keyed_vector.vocab
    keyed_vector.index2word = old_keyed_vector.index2word
    keyed_vector.vectors = new_matrix
    assert (len(old_keyed_vector.vocab), vector_size) == keyed_vector.vectors.shape
    return keyed_vector
