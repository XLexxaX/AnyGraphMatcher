import logging
import os
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from eval.createLexicon import get_lexicon_keyed_vectors, write_lexicon_to_file

logger = logging.getLogger(__name__)
package_directory = os.path.dirname(os.path.abspath(__file__))

def create_lexicon():
    source_path = os.path.join(package_directory, '..', 'data', 'small',
                               'test.txt')  # os.path.join(package_directory, '..', 'data', 'wmt11', 'training-monolingual', 'news.2011.en.shuffled.tokenized.lowercased.unique')
    target_path = os.path.join(package_directory, '..', 'data', 'small', 'test_two.txt')
    # target_sentences = os.path.join(package_directory, '..', 'data', 'wmt11', 'training-monolingual', 'news.2011.fr.shuffled')

    src = Word2Vec(LineSentence(source_path), min_count=0)
    target = Word2Vec(LineSentence(target_path), min_count=0)

    lexicon = get_lexicon_keyed_vectors(src.wv, target.wv, 'de', 'en', 3)
    write_lexicon_to_file(lexicon, 'my_lexicon.csv')


from eval.evalnlp import evaluate



def first_system(src, tgt, initial_mapping):
    from crosslingual.randomtranslation import get_random_translation_embedding
    from wordembedding.word2vec import word2vec_embedding_from_sentences_func
    sg = 0
    size = 100
    window = 5

    #src_model = Word2Vec(LineSentence(src), sg=sg, size=size, window=window, workers=1, sample=0, min_count=0)
    #tgt_model = Word2Vec(LineSentence(tgt), sg=sg, size=size, window=window, workers=1, sample=0, min_count=0)

    embedding = get_random_translation_embedding(LineSentence(src), LineSentence(tgt), initial_mapping, word2vec_embedding_from_sentences_func())





def run():
    evaluate(first_system, [('en', 'fr')])


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
    logging.info("Start")
    run()
