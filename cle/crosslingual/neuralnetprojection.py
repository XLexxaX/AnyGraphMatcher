from crosslingual.util import project_embeddings_to_lexicon_subset, create_keyed_vector

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras import losses
from sklearn.model_selection import KFold, ShuffleSplit

def neural_net_projection(word_vector_src, word_vector_tgt, lexicon):
    matrix_src, matrix_tgt = project_embeddings_to_lexicon_subset(word_vector_src, word_vector_tgt, lexicon)

    model = Sequential()
    model.add(Dense(word_vector_src.vector_size, input_dim=word_vector_src.vector_size, activation='relu'))
    #model.add(Dropout(0.5))

    #model.add(Dense(word_vector_tgt.vector_size, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=losses.mean_squared_error,
                  optimizer=sgd,
                  metrics=[losses.mean_squared_error])

    model.fit(matrix_src, matrix_tgt, epochs=2000, batch_size=128)

    source_projected = model.predict(word_vector_src.vectors)
    source_projected_keyed_vector = create_keyed_vector(word_vector_src, source_projected)
    return source_projected_keyed_vector
    # cross_val = ShuffleSplit(n_splits=1, test_size=0.33)#KFold(n_splits=2)
    # for train_indices, test_indices in cross_val.split(matrix_src):
    #     src_train, dst_train = matrix_src[train_indices], matrix_tgt[train_indices]
    #     src_test, dst_test = matrix_src[test_indices], matrix_tgt[test_indices]
    #
    #     model.fit(src_train, dst_train,epochs=20, batch_size=128)
    #     score = model.evaluate(src_test, dst_test, batch_size=128)



    #source_projected = create_keyed_vector(word_vector_src, np.dot(word_vector_src.wv.vectors, w))
    #return source_projected

