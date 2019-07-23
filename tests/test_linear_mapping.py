from cle.crosslingual.neuralnetprojection import neural_net_projection
from cle.crosslingual.linearprojection import linear_projection, orth_projection#, #sgd_projection
from gensim.models.keyedvectors import Vocab
from gensim.models import KeyedVectors
import numpy as np

def __create_random_keyed_vector(matrix):
    vocab = dict()
    index_to_word = []
    for word_id in range(matrix.shape[0]):
        index_to_word.append(str(word_id))
        vocab[str(word_id)] = Vocab(index=word_id, count=2)
    vector_size = matrix.shape[1]

    keyed_vector = KeyedVectors(vector_size)
    keyed_vector.vector_size = vector_size
    keyed_vector.vocab = vocab
    keyed_vector.index2word = index_to_word
    keyed_vector.vectors = matrix
    return keyed_vector



def __create_test_case(number_of_instances, number_of_mappings, vector_size):
    source = np.random.rand(number_of_instances, vector_size)
    true_linear_mapping = np.random.rand(1, vector_size)
    target = source * true_linear_mapping

    source_vector = __create_random_keyed_vector(source)
    target_vector = __create_random_keyed_vector(target)

    lexicon = [(str(i),str(i)) for i in range(number_of_mappings) ]

    return source_vector, target_vector, lexicon


# def test_neural_net():
#     source_vector, target_vector, lexicon = __create_test_case(500,20,50)
#
#     source_projected = neural_net_projection(source_vector, target_vector, lexicon)
#
#     test = source_projected.vectors[50:, :]
#     gold = target_vector.vectors[50:, :]
#     mse = ((gold - test) ** 2).mean()
#     print(mse)

#test = source_projected.vectors[50:, :]
#gold = target_vector.vectors[50:, :]
#mse = ((gold - test) ** 2).mean()
#print(mse)
#
def test_linear_projection():
     source_vector, target_vector, lexicon = __create_test_case(500,250,50)
     source_projected = linear_projection(source_vector, target_vector, lexicon)
     mse = ((source_projected.vectors - target_vector.vectors) ** 2).mean()
     np.set_printoptions(precision=6)
     np.set_printoptions(suppress=True)
     print('linear_projection: ' + str(np.array([mse])))
     print(mse)

# def test_sgd_projection():
#     source_vector, target_vector, lexicon = __create_test_case(500, 250, 50)
#     source_projected = sgd_projection(source_vector, target_vector, lexicon)
#     mse = ((source_projected.vectors - target_vector.vectors) ** 2).mean()
#     print(mse)
#
# def test_orth_projection():
#     source_vector, target_vector, lexicon = __create_test_case(500, 250, 50)
#     source_projected = orth_projection(source_vector, target_vector, lexicon)
#     mse = ((source_projected.vectors - target_vector.vectors) ** 2).mean()
#     print(mse)

# def test_neural_net():
#     source_vector, target_vector, lexicon = __create_test_case(500, 250, 50)
#     source_projected = neural_net_projection(source_vector, target_vector, lexicon)
#     mse = ((source_projected.vectors - target_vector.vectors) ** 2).mean()
#     print(mse)
#
# def test_baseline():
#     source_vector, target_vector, lexicon = __create_test_case(500, 250, 50)
#     source_projected = source_vector
#     mse = ((source_projected.vectors - target_vector.vectors) ** 2).mean()
#     print(mse)