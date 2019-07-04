import logging
from cle.matcher.basematcher import BaseMatcher
from cle.crosslingual.randomtranslation import get_random_translation_embedding

logger = logging.getLogger(__name__)

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

class RandomTranslationMatcher(BaseMatcher):

    def __init__(self, embedding_generation_function, sentence_generator_function, **args):
        self.embedding_generation_function = embedding_generation_function
        self.sentence_generator_function = sentence_generator_function
        self.args = args


    def compute_mapping(self):
        self.cross_lingual_embedding = get_random_translation_embedding(
            self.sentence_generator_function(self.src_kg), self.sentence_generator_function(self.dst_kg),
            self.generate_lexicon_from_initial_mapping(), self.embedding_generation_function,
            **self.args
        )

    def get_mapping(self):
        mapping = set()
        # with open('out.txt', 'w', encoding='utf-8') as f:
        #     for sentence in generate_random_translation(
        #             self.sentence_generator_function(self.src_kg),self.sentence_generator_function(self.dst_kg),
        #             self.generate_lexicon_from_initial_mapping(), **self.args):
        #         f.write(' '.join(sentence) + '\n')
        #
        # model = Word2Vec(LineSentence('out.txt'), size=100)
        #
        # from cle.wordembedding.visualiseword import vis_matplot
        #
        # vis_matplot(model)

        #test = self.cross_lingual_embedding
        return mapping

    def get_mapping_with_ranking(self, elements, topn=20):
        results = []
        for e in elements:
            if e not in self.cross_lingual_embedding:
                results.append([])
            else:
                results.append(self.cross_lingual_embedding.most_similar(e, topn=topn))
        return results
