import logging
from cle.matcher.basematcher import BaseMatcher
from cle.crosslingual.ccaprojection import cca_projection

logger = logging.getLogger(__name__)


class CCAMatcher(BaseMatcher):

    def __init__(self, embedding_generation_function, top_correlation_ratio=0.5):
        self.embedding_generation_function = embedding_generation_function
        self.top_correlation_ratio = top_correlation_ratio
        self.src_proj_embedding = None
        self.dst_proj_embedding = None

    def compute_mapping(self):
        src_embedding = self.embedding_generation_function(self.src_kg)
        dst_embedding = self.embedding_generation_function(self.dst_kg)

        self.src_proj_embedding, self.dst_proj_embedding = cca_projection(
            src_embedding, dst_embedding, self.generate_lexicon_from_initial_mapping(), self.top_correlation_ratio)

    def get_mapping(self):
        mapping = set()

        return mapping

    def get_mapping_with_ranking(self, elements, topn=20):
        results = []
        for e in elements:
            if e not in self.src_proj_embedding:
                results.append([])
            else:
                results.append(self.dst_proj_embedding.most_similar(positive=[self.src_proj_embedding[e]], topn=topn))
        return results
