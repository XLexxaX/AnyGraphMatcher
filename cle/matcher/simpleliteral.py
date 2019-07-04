from cle.tokenization.simpletoken import tokenize
from cle.matcher.basematcher import BaseMatcher


class SimpleLiteralMatcher(BaseMatcher):

    def compute_mapping(self):
        self.mapping = set()
        self.mapping_dict = dict()
        labels_source = dict()
        for s, p, o in self.src_kg.get_literal_triples():
            if p == 'http://www.w3.org/2000/01/rdf-schema#label' or p == 'rdfs:label':
                labels_source[' '.join(tokenize(o))] = s

        for s, p, o in self.dst_kg.get_literal_triples():
            if p == 'http://www.w3.org/2000/01/rdf-schema#label' or p == 'rdfs:label':
                source_resource = labels_source.get(' '.join(tokenize(o)))
                if source_resource is not None:
                    self.mapping.add((source_resource, s, '=', 1.0))
                    self.mapping_dict[source_resource] = [(s, 1.0)]

    def get_mapping(self):
        return self.mapping

    def get_mapping_with_ranking(self, elements, topn=20):
        results = []
        for e in elements:
            results.append(self.mapping_dict.get(e, []))
        return results



# def get_literal_mappings(kg_src, kg_dst, use_fragment=False, use_label=False, use_comment=False):
#     mapping = set()
#
#     labels_source = dict()
#     for s,p,o in kg_src.get_literal_triples():
#         if p == 'http://www.w3.org/2000/01/rdf-schema#label' or p == 'rdfs:label':
#             labels_source[' '.join(tokenize(o))] = s
#
#     for s,p,o in kg_dst.get_literal_triples():
#         if p == 'http://www.w3.org/2000/01/rdf-schema#label' or p == 'rdfs:label':
#             source_resource = labels_source.get(' '.join(tokenize(o)))
#             if source_resource is not None:
#                 mapping.add((source_resource, s, '=', 1.0))
#
#     return mapping