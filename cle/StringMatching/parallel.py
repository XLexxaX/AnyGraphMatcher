from cle.StringMatching.GraphToolbox import GraphManager
import operator
import random
import multiprocessing as mp
import cle.StringMatching.StringMatcher_Interface as smi


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

    def preciseBatchMatch(self, path, min_similarity, index_properties, src_triples, tgt_triples, max_similarity=1.0):
        correspondences = dict()
        i=0
        total_size = len(self.src_graphmanager.graph.keys())
        current_run = 1
        allkeys = list()
        proccount=mp.cpu_count()
        partkeys = list()
        for nodeid in self.src_graphmanager.graph.keys():
            i=i+1
            if i < total_size/proccount*current_run:
                partkeys.append(nodeid)
            else:
                partkeys.append(nodeid)
                current_run = current_run + 1
                allkeys.append(partkeys)
                partkeys = list()
        jobs = list()
        queues = list()
        x=0
        i=0
        parent_conn, child_conn = mp.Pipe()
        for keys in allkeys:
            x=x+1
            ctx = mp.get_context('spawn')
            q = ctx.Queue()
            process = ctx.Process(target=smi.main, args=(keys, q, x, child_conn, index_properties, src_triples, tgt_triples))
            jobs.append(process)
            queues.append(q)
        for j in jobs:
            j.start()
        while i < total_size-1:
            i = i + int(parent_conn.recv())
            print('         Blocking by syntax, progress: ' + str(int(100 * i / (total_size))) + '%', end="\r")
        for q in queues:
            f = open(path, 'a+')
            f.write(str(q.get()))
            f.close()
        for j in jobs:
            j.join()
        child_conn.close()
        return None





def main(src_triples, tgt_triples, labels, filename):

    # match products
    if type(labels) == str:
        index_properties = smi.get_labels_from_file(labels)
    else:
        index_properties = labels
    assert type(index_properties) == list, "Labels-parameter must be provided as a list of label-properties"
    #print('label size: ' + str(len(index_properties)))


    stringmatcher = StringMatcher(src_triples,tgt_triples, index_properties)
    mappings = stringmatcher.preciseBatchMatch(filename, 0.9, index_properties, src_triples, tgt_triples)
    #pprint.PrettyPrinter().pprint(str(mappings))
    #formatted_save(mappings, "/sapmnt/home/D072202/RData2Graph/rdata2graph/data/sap_hilti_data/sap_hilti_gold.csv")

    # match categories
    #index_properties = ["<http://rdata2graph.sap.com/hilti_erp/property/T179.Description>", "<http://rdata2graph.sap.com/hilti_web/property/Categories.name>"]
    #stringmatcher = StringMatcher("C:/Users/D072202/RData2Graph/rdata2graph/data/sap_hilti_data/graph_triples_hilti_erp.nt","C:/Users/D072202/RData2Graph/rdata2graph/data/sap_hilti_data/graph_triples_hilti_web.nt")
    #mappings = stringmatcher.preciseBatchMatch(0.85)
    #pprint.PrettyPrinter().pprint(str(mappings))

    #formatted_save(mappings, "C:/Users/D072202/RData2Graph/rdata2graph/data/sap_hilti_data/sap_hilti_gold.csv")
