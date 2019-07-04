from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

#from gensim.models import KeyedVectors
#KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
__color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def __plot_tsne(array, vocab, color=None):
    tsne = TSNE(n_components=2, method='exact')#, random_state=0)
    #np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(array)
    plt.scatter(Y[:, 0], Y[:, 1], c=color)
    for label, x, y in zip(vocab, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()


def __plot_tsne2(array, vocab, color=None):
    array.index = range(len(array.index))
    vocab.index = range(len(vocab.index))
    vocab = vocab.values
    tsne = TSNE(n_components=2, method='exact')#, random_state=0)
    #np.set_printoptions(suppress=True)
    print("Performing TSNE")
    Y = tsne.fit_transform(array)
    print("TSNE done")
    Y = Y[:10, :]
    plt.scatter(Y[:, 0], Y[:, 1], c=color)
    for label, x, y in zip(vocab, Y[:, 0], Y[:, 1]):
        if (not label == "" and label is not None):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()

def vis_matplot(gensim_w2v_model, number_of_words=100):
    __plot_tsne(
        gensim_w2v_model[gensim_w2v_model.wv.vocab][:number_of_words, :],
        gensim_w2v_model.wv.vocab
    )


def vis_matplot_similar(keyed_vector, *args, amount=10):
    number_of_elements = len(args) + (len(args) * amount)
    arr = np.zeros((number_of_elements, keyed_vector.vector_size))
    vocab = []
    i = 0
    colors = []
    color = 0
    for root in args:
        arr[i, :] = keyed_vector[root]
        vocab.append(root)
        i += 1

        for word, score in keyed_vector.most_similar(root):
            arr[i, :] = keyed_vector[word]
            vocab.append(word)
            i += 1
            colors.append(__color_list[color])#
        color += 1

    __plot_tsne(arr, vocab, colors)


if __name__ == '__main__':

    import pandas as pd
    from cle.StringMatching.GraphToolbox import GraphManager
    #trp_path = "/sapmnt/home/D072202/Dev/CrossLingualEmbeddingsForMatching-master/data/sap_hilti_data/sap_hilti_data/graph_triples_hilti_erp.nt"
    trp_path = "/sapmnt/home/D072202/Dev/CrossLingualEmbeddingsForMatching-master/data/sap_hilti_data/sap_hilti_data/graph_triples_hilti_web.nt"
    src_graphmanager = GraphManager()
    src_graphmanager.setIndexProperties([])
    src_graphmanager.readGraphFromNTFile(trp_path)


    src_path = "/sapmnt/home/D072202/Dev/CrossLingualEmbeddingsForMatching-master/cle/crosslingual/muse/MUSE/data/embeddings2.vec"
    src = pd.read_csv(src_path, delimiter=" ", header=None, skiprows=1)
    src.columns = ["src_id"] + ["src_" + str(i) for i in range(len(src.columns)-1)]
    src = src.dropna(axis=1)
    src = src.sample(n=20)
    from sklearn.utils import shuffle
    src = shuffle(src)
    src.reset_index()
    data = pd.DataFrame([])
    for index, row in src.iterrows():
        try:
            row['label'] = src_graphmanager.graph[row['src_id']]["<http://rdata2graph.sap.com/hiti_web/property/Products.Name>"]#["<http://rdata2graph.sap.com/hilti_erp/property/T179.Description>"]
            data = data.append(row)
        except:
            pass
    __plot_tsne2(data.loc[:,["src_" + str(i) for i in range(len(src.columns)-1)]],
                data.loc[:,['label']])

# import tensorflow as tf
# from tensorflow.contrib.tensorboard.plugins import projector
# import numpy as np
# import logging
#
# def vis_tensorboard(gensim_w2v_model):
#     max_size = len(gensim_w2v_model.wv.vocab) - 1
#     w2v = np.zeros((max_size, gensim_w2v_model.wv.vector_size))
#
#     with open("metadata.tsv", 'w', encoding='utf-8') as file_metadata:
#         for i, word in enumerate(gensim_w2v_model.wv.index2word[:max_size]):
#             w2v[i] = gensim_w2v_model.wv[word]
#             file_metadata.write(word + '\n')
#
#     logging.info("ONE")
#     sess = tf.InteractiveSession()
#     embedding = tf.Variable(w2v, trainable=False, name='embedding')
#     logging.info("TWO")
#     tf.global_variables_initializer().run()
#     saver = tf.train.Saver()
#     logging.info("Three")
#     writer = tf.summary.FileWriter('tensorboard', sess.graph)
#
#     logging.info("four")
#
#     # adding into projector
#     config = projector.ProjectorConfig()
#     embed = config.embeddings.add()
#     embed.tensor_name = 'embedding'
#     embed.metadata_path = 'metadata.tsv'
#
#     projector.visualize_embeddings(writer, config)
#     saver.save(sess, 'session.ckpt', global_step=max_size)
