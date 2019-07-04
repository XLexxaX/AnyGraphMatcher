from cle.sentencegenerator.saveutil import save_sentences
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def word2vec_embedding_from_sentences(sentence_generator, sg=0, size=100, window=5):
    # TODO: from tempfile import TemporaryFile ? https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html
    with open('tmp.txt', 'w', encoding='utf-8') as f:
        for sentence in sentence_generator:
            f.write(' '.join(sentence) + '\n')
    #save_sentences(sentence_generator, 'tmp.txt')
    model = Word2Vec(LineSentence('tmp.txt'), sg=sg, size=size, window=window, workers=1, sample=0, min_count=0)
    return model.wv

def word2vec_embedding_from_sentences_func(sg=0, size=100, window=5):
    return lambda sentence_generator: word2vec_embedding_from_sentences(sentence_generator, sg, size, window)

def word2vec_embedding_from_kg(sentence_generator, sg=0, size=100, window=5):
    return lambda kg: word2vec_embedding_from_sentences(sentence_generator(kg), sg, size, window)

def word2vec_embedding_from_sentences_v2(sentences, CONFIGURATION, sg=0, size=100, window=100):

    from cle.wordembedding import EmbeddingHelper
    model = EmbeddingHelper.embed(sentences, size, CONFIGURATION, window=100)
    return model.wv
