import logging

from eval.evaldbkwik import evaluate
from sentencegenerator.randomwalks import generate_random_walks
from sentencegenerator.literalwalks import generate_literal_walks
from wordembedding.word2vec import word2vec_embedding_from_sentences, word2vec_embedding_from_kg
from matcher.linearprojectionmatcher import LinearProjectionMatcher
from matcher.analysematcher import AnalyseMatcher
from matcher.sharedembeddingmatcher import SharedEmbeddingMatcher
from matcher.simpleliteral import SimpleLiteralMatcher
from matcher.randtransmatcher import RandomTranslationMatcher
from crosslingual.linearprojection import linear_projection# ,sgd_projection, orth_projection
from matcher.ccamatcher import CCAMatcher
from kgembedding.docembedding import doc2vec_embedding, doc2vec_embedding_from_kg

from itertools import chain

logger = logging.getLogger(__name__)

from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
    logging.info("Start")

    #linear projection for classes, properties, instances?


    # evaluate(
    #     SimpleLiteralMatcher(),
    #     'dbkwik', selection_sub_tracks=set(['darkscape-oldschoolrunescape']))

    evaluate(
        AnalyseMatcher(),
        'dbkwik', selection_sub_tracks=set(['darkscape-oldschoolrunescape']))

    evaluate(
       LinearProjectionMatcher(word2vec_embedding_from_kg(generate_random_walks), orth_projection),
       'dbkwik', selection_sub_tracks=set(['darkscape-oldschoolrunescape']), initial_mapping_system=SimpleLiteralMatcher())
        #inital_mapping_gold_standard=True,percentage_initial_mapping=1.0)

    # evaluate(
    #     LinearProjectionMatcher(word2vec_embedding_from_kg(generate_random_walks), linear_projection),
    #     'dbkwik', selection_sub_tracks=set(['darkscape-oldschoolrunescape']), inital_mapping_gold_standard=True,
    #     percentage_initial_mapping=1.0)
    #
    # evaluate(
    #     RandomTranslationMatcher(word2vec_embedding_from_sentences, generate_random_walks, replacement_chance=0.5, balance_and_randomize=False),
    #     'dbkwik', selection_sub_tracks=set(['darkscape-oldschoolrunescape']), initial_mapping_system=SimpleLiteralMatcher())
    #
    # evaluate(
    #     CCAMatcher(word2vec_embedding_from_kg(generate_random_walks, size=53), top_correlation_ratio=1.0),
    #     'dbkwik', selection_sub_tracks=set(['darkscape-oldschoolrunescape']), initial_mapping_system=SimpleLiteralMatcher())
    #     #'dbkwik',selection_sub_tracks=set(['darkscape-oldschoolrunescape']), inital_mapping_gold_standard=True, percentage_initial_mapping=1.0)

    # evaluate(
    #     LinearProjectionMatcher(word2vec_embedding_from_kg(lambda kg: chain(generate_random_walks(kg), generate_literal_walks_with_text(kg))), linear_projection),
    #     'conference_conference-v1', selection_sub_tracks=set(['cmt-conference']),
    #     inital_mapping_gold_standard=True, percentage_initial_mapping=1.0)

    #evaluate(
    #    #SharedEmbeddingMatcher(word2vec_embedding_from_kg(lambda kg: generate_literal_walks(kg))),  # lambda kg: generate_literal_walks(kg, text_link_amount=1)     lambda kg: LineSentence('conference_literals.txt')
    #    SharedEmbeddingMatcher(doc2vec_embedding_from_kg()),
    #    'conference_conference-v1', selection_sub_tracks=set(['cmt-conference']),
    #    inital_mapping_gold_standard=True, percentage_initial_mapping=1.0)


    # from gensim.test.utils import common_texts
    # from wordembedding.lsa import lsa_embedding_from_sentences
    #
    # CONFIGURATION.log(lsa_embedding_from_sentences(common_texts, size=10))


#############
#SimpleLiteralMatcher
# 2018-08-20 20:19:59,140 INFO:                 Classes                  |                Properties                 |                 Instances
# 2018-08-20 20:19:59,140 INFO: Prec   Rec    F-1    H@1    H@5    H@10  |  Prec   Rec    F-1    H@1    H@5    H@10  |  Prec   Rec    F-1    H@1    H@5    H@10
# 2018-08-20 20:19:59,140 INFO: 71.43  45.45  55.56   9.09   9.09   9.09 |  88.89  57.14  69.57  42.86  42.86  42.86 |  44.00  95.65  60.27  32.61  32.61  32.61
# 2018-08-20 20:22:59,077 INFO:  4.00   9.09   5.56  45.45  45.45  45.45 |  33.33   7.14  11.76   7.14   7.14   7.14 |   1.45   2.17   1.74   2.17   2.17   2.17   #linear projection
# 2018-08-20 20:24:47,309 INFO:  0.00   0.00   0.00   0.00   0.00   0.00 |   0.00   0.00   0.00   0.00   0.00   0.00 |   0.00   0.00   0.00   0.00   0.00   0.00   #sgd
# 2018-08-20 20:29:16,792 INFO:  5.56   9.09   6.90  36.36  36.36  36.36 |  50.00   7.14  12.50   7.14   7.14   7.14 |   0.00   0.00   0.00   0.00   0.00   0.00   # orth
# 2018-08-20 20:30:31,392 INFO:  0.00   0.00   0.00  18.18  18.18  18.18 |   0.00   0.00   0.00   0.00   0.00   0.00 |   0.00   0.00   0.00   2.17   2.17   2.17   # random translation
# 2018-08-20 20:32:05,557 INFO:  0.00   0.00   0.00  27.27  27.27  27.27 |   0.00   0.00   0.00   0.00   0.00   0.00 |   0.00   0.00   0.00   0.00   0.00   0.00   # cca