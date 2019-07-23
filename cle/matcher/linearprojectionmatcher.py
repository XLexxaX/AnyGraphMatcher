import logging
from matcher.basematcher import BaseMatcher


logger = logging.getLogger(__name__)


class LinearProjectionMatcher(BaseMatcher):

    def __init__(self, embedding_generation_function, linear_projection_function):
        self.embedding_generation_function = embedding_generation_function
        self.linear_projection_function = linear_projection_function
        self.src_proj_embedding = None
        self.dst_embedding = None

    def compute_mapping(self):
        src_embedding = self.embedding_generation_function(self.src_kg)
        self.dst_embedding = self.embedding_generation_function(self.dst_kg)
        self.src_proj_embedding = self.linear_projection_function(
            src_embedding, self.dst_embedding, self.generate_lexicon_from_initial_mapping()
        )

    def get_mapping(self):
        mapping = set()

        #iterate over the embedding matrix
        for src_element, vocab in self.src_proj_embedding.vocab.items():
            dst_most_similar = self.dst_embedding.most_similar(positive=[self.src_proj_embedding[src_element]], topn=5)
            for dst_element, confidence in dst_most_similar:
                #if dst_element == class:
                if confidence > 0.90:
                    mapping.add((src_element, dst_element, '=', confidence))
                    break

            #print(src_element)


        # from_training = []
        # for src, dst, rel, confidence in self.initial_mapping:
        #     from_training.append(src)
        #
        # self.get_mapping_with_ranking(from_training)#['http://dbkwik.webdatacommons.org/darkscape/resource/Soaked_kindling'])

        #find simmilar between src_proj_embedding and dst_embedding

        return mapping

    def get_mapping_with_ranking(self, elements, topn=20):
        results = []
        for e in elements:
            if e not in self.src_proj_embedding:
                results.append([])
            else:
                results.append(self.dst_embedding.most_similar(positive=[self.src_proj_embedding[e]], topn=topn))
        return results
