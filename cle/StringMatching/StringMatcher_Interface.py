from StringMatching.GraphToolbox import GraphManager
from StringMatching.InvertedIndexToolbox import getNGrams
import operator
import pprint
import random
import heapq
import os

class StringMatcher:

    def __init__(self, src_file_path, tgt_file_path, index_properties):
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

    def preciseBatchMatch(self, min_similarity, progressqueue, max_similarity=1.0, keys=None, x=0):
        #print('Starting matching')
        #f = open(path+str(x), "a+", encoding="UTF-8")
        correspondences = ""
        i=0
        total_size = len(keys)
        for nodeid in keys:
            i=i+1
            #print('         Blocking by syntax, progress: ' + str(int(100 * i / (total_size))) + '%', end="\r")
            progressqueue.send(1)
            if random.randint(1,101) > 100:
                continue
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
                pass
                #correspondences[nodeid] = None
            else:
                best_matching_resources = heapq.nlargest(20,tmp_tgt_ind.items(), key=operator.itemgetter(1)) #max((tmp_tgt_ind.items()), key=operator.itemgetter(1))
                for best_matching_resource in best_matching_resources:#max((tmp_tgt_ind.items()), key=operator.itemgetter(1))
                    maximum_ngrams_x = 0
                    maximum_ngrams_y = 0
                    for src_prop in self.src_graphmanager.indices.keys():
                        try:
                            maximum_ngrams_x = maximum_ngrams_x + len(getNGrams(self.src_graphmanager.graph[nodeid][src_prop]))
                        except KeyError:
                            pass
                    for tgt_prop in self.tgt_graphmanager.indices.keys():
                        try:
                            maximum_ngrams_y = maximum_ngrams_y + len(self.tgt_graphmanager.graph[best_matching_resource[0]][tgt_prop])
                        except KeyError:
                            pass
                    no_of_ngrams = max(maximum_ngrams_x, maximum_ngrams_y)
                    if no_of_ngrams > 1 and best_matching_resource[1] > no_of_ngrams*min_similarity and best_matching_resource[1] < no_of_ngrams and best_matching_resource[1] <= min(maximum_ngrams_x, maximum_ngrams_y)*max_similarity:
                        try:
                                #correspondences[nodeid] = best_matching_resource[0]
                                #l = self.src_graphmanager.graph[nodeid]["<http://rdata2graph.sap.com/darkscape/non-player_character.label>".lower()] + " -> " + self.tgt_graphmanager.graph[best_matching_resource[0]]["<http://rdata2graph.sap.com/oldschoolrunescape/non-player_character.label>".lower()] + " <----> " + str(nodeid).replace("<","").replace(">","") + "\t" + str(best_matching_resource[0]).replace("<","").replace(">","") + "\r\n"
                                l = str(nodeid).replace("<","").replace(">","") + "\t" + str(best_matching_resource[0]).replace("<","").replace(">","") +'\t'+ str('1')+ "\r\n"
                                if not correspondences == "":
                                    correspondences = correspondences
                                correspondences = correspondences + l.lower()
                                #f.write(l.lower())
                                #f.flush()
                        except KeyError:
                            pass
                    #else:
                    #    correspondences[nodeid] = None
        #f.close()
        return correspondences




def main(keys, q, x, progressqueue, index_properties, src_triples, tgt_triples):

    stringmatcher = StringMatcher(src_triples,tgt_triples, index_properties)
    mappings = stringmatcher.preciseBatchMatch(0.3, progressqueue, 1.0, keys, x)
    q.put(mappings)



def get_labels_from_file(path):
    index_properties = list()
    with open(path, mode="r", encoding="UTF-8") as f:
        for label in f:
            if random.randint(1,101) > 0:
                index_properties.append(label.replace('\n',''))
    return index_properties
