from StringMatching.GraphToolbox import GraphManager
from StringMatching.InvertedIndexToolbox import getNGrams
import operator
import pprint

class StringMatcher:

    def __init__(self, src_file_path, tgt_file_path):
        self.src_graphmanager = GraphManager()
        self.src_graphmanager.setIndexProperties(index_properties)
        self.src_graphmanager.readGraphFromNTFile(src_file_path)
        self.tgt_graphmanager = GraphManager()
        self.tgt_graphmanager.setIndexProperties(index_properties)
        self.tgt_graphmanager.readGraphFromNTFile(tgt_file_path)

    def batchMatch(self):
        correspondences = dict()
        for nodeid in self.src_graphmanager.graph.keys():
            indices = []
            for src_prop in self.src_graphmanager.indices.keys():
                for tgt_prop in self.tgt_graphmanager.indices.keys():
                    try:
                        indices = indices + self.tgt_graphmanager.indices[tgt_prop].getIndicesForValue(self.src_graphmanager.graph[nodeid][src_prop])
                    except KeyError:
                        pass
            tmp_tgt_ind = dict()
            for index in indices:
                if index in tmp_tgt_ind.keys():
                    tmp_tgt_ind[index] = tmp_tgt_ind[index] + 1
                else:
                    tmp_tgt_ind[index] = 1
            if len(tmp_tgt_ind)<1:
                correspondences[nodeid] = None
            else:
                correspondences[nodeid] = max((tmp_tgt_ind.items()), key=operator.itemgetter(1))[0]
        return correspondences

    def preciseBatchMatch(self, min_similarity):
        correspondences = dict()
        for nodeid in self.src_graphmanager.graph.keys():
            indices = []
            for src_prop in self.src_graphmanager.indices.keys():
                for tgt_prop in self.tgt_graphmanager.indices.keys():
                    try:
                        indices = indices + self.tgt_graphmanager.indices[tgt_prop].getIndicesForValue(self.src_graphmanager.graph[nodeid][src_prop])
                    except KeyError:
                        pass
            tmp_tgt_ind = dict()
            for index in indices:
                if index in tmp_tgt_ind.keys():
                    tmp_tgt_ind[index] = tmp_tgt_ind[index] + 1
                else:
                    tmp_tgt_ind[index] = 1
            if len(tmp_tgt_ind)<1:
                correspondences[nodeid] = None
            else:
                best_matching_resource = max((tmp_tgt_ind.items()), key=operator.itemgetter(1))
                for src_prop in self.src_graphmanager.graph[nodeid].keys():
                    for tgt_prop in self.tgt_graphmanager.graph[best_matching_resource[0]].keys():
                        no_of_ngrams = min(len(getNGrams(self.src_graphmanager.graph[nodeid][src_prop])),
                                           len(self.tgt_graphmanager.graph[best_matching_resource[0]][tgt_prop]))
                if best_matching_resource[1] > no_of_ngrams*min_similarity:
                    correspondences[nodeid] = best_matching_resource[0]
                else:
                    correspondences[nodeid] = None
        return correspondences


if __name__ == '__main__':
    index_properties = ["<http://rdata2graph.sap.com/hilti_erp/property/T179.Description>", "<http://rdata2graph.sap.com/hiti_web/property/Products.Name>"]
    stringmatcher = StringMatcher("C:/Users/D072202/RData2Graph/rdata2graph/data/sap_hilti_data/graph_triples_hilti_erp.nt","C:/Users/D072202/RData2Graph/rdata2graph/data/sap_hilti_data/graph_triples_hiti_web.nt")
    pprint.PrettyPrinter().pprint(str(stringmatcher.preciseBatchMatch(0.95)))
