import logging
from cle.matcher.basematcher import BaseMatcher
from cle.loadkg.kg import CombinedKG

logger = logging.getLogger(__name__)

from cle.wordembedding.visualise import vis_matplot, vis_matplot_similar

class SharedEmbeddingMatcher(BaseMatcher):

    def __init__(self, embedding_generation_function):
        self.embedding_generation_function = embedding_generation_function
        self.shared_embedding = None

    def compute_mapping(self):
        self.shared_embedding = self.embedding_generation_function(CombinedKG(self.src_kg, self.dst_kg))

    def get_mapping(self):
        mapping = set()

        test = self.shared_embedding.vocab

        vis_matplot_similar(self.shared_embedding, 'http://cmt#Person', 'http://conference#Person')

        #iterate over the embedding matrix
        for element, vocab in self.shared_embedding.vocab.items():
            nearest_element, confidence = self.shared_embedding.most_similar(element, topn=1)[0]#, topn=5)
            if confidence > 0.9:
                mapping.add((element, nearest_element, '=', confidence))
            #print("test")
            # for dst_element, confidence in dst_most_similar:
            #     #if dst_element == class:
            #     if confidence > 0.90:
            #         mapping.add((src_element, dst_element, '=', confidence))
            #         break

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
            if e not in self.shared_embedding:
                results.append([])
            else:
                results.append(self.shared_embedding.most_similar(e, topn=topn))
        return results
