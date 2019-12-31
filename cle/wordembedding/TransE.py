#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#from nltk.tokenize import word_tokenize
from configurations.PipelineTools import PipelineDataTuple
import numpy as np
import pandas as pd
from wordembedding.TransE_modules.prepare_pretrained import load as prepare_pretrained
from wordembedding.TransE_modules.Train import trainTransE as trainTransE

import os
import pickle
import sys

HETEROGENEITY_THRESHOLD = 0.75
global CONFIGURATION

def interface(main_input, args, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    graph1 = main_input.get(0)
    graph2 = main_input.get(1)
    dim = args.get(0)
    assert graph1 is not None, "Graph1 not found in " + os.path.basename(sys.argv[0])
    assert graph2 is not None, "Graph2 not found in " + os.path.basename(sys.argv[0])
    assert dim is not None, "Dimension not found in " + os.path.basename(sys.argv[0])
    execute(dim, graph1, graph2)
    return PipelineDataTuple(graph1, graph2)

def fill_graph(rundir, graph1, graph2):

        original_embeddings_dir = rundir + "data\\FB15K\\"

        entityInput = open(rundir + "data\\outputData\\entity2vec.pickle", "rb")
        relationInput = open(rundir + "data\\outputData\\relation2vec.pickle", "rb")
        gpuEntityEmbeddings = pickle.load(entityInput)
        gpuRelationEmbeddings = pickle.load(relationInput)
        entityInput.close()
        relationInput.close()
        entity_embeddings = gpuEntityEmbeddings.cpu()
        relation_embeddings = gpuRelationEmbeddings.cpu()

        e_idxlkp = pd.read_csv(original_embeddings_dir + "entity_embeddings.nt", encoding="UTF-8",header=None, sep="\t", index_col=False)
        e_idxlkp = e_idxlkp.set_index([0])
        e_idxlkp = e_idxlkp.loc[:,:1]
        r_idxlkp = pd.read_csv(original_embeddings_dir + "relation_embeddings.nt", encoding="UTF-8",header=None, sep="\t", index_col=False)
        r_idxlkp = r_idxlkp.set_index([0])
        r_idxlkp = r_idxlkp.loc[:,:1]

        for descriptor, resource in graph1.elements.items():
            try:
                if descriptor in e_idxlkp.index:
                    test = entity_embeddings[[int(e_idxlkp.loc["s/<"+descriptor+">", 1])]]
                elif descriptor in r_idxlkp.index:
                    test = relation_embeddings[[int(r_idxlkp.loc["p/<"+descriptor+">", 1])]]
                else:
                    raise KeyError('')
                test = np.array(test).astype(float).tolist()
                resource.embeddings = [test]
            except KeyError:
                #Embedidng already defined
                CONFIGURATION.log("Key " + descriptor + " not found ... proceeding\n")

        for descriptor, resource in graph2.elements.items():
            try:
                if descriptor in e_idxlkp.index:
                    test = entity_embeddings[[int(e_idxlkp.loc["s/<"+descriptor+">", 1])]]
                elif descriptor in r_idxlkp.index:
                    test = relation_embeddings[[int(r_idxlkp.loc["p/<"+descriptor+">", 1])]]
                else:
                    raise KeyError('')
                test = np.array(test).astype(float).tolist()
                resource.embeddings = [test]
            except KeyError:
                #Embedidng already defined
                CONFIGURATION.log("Key " + descriptor + " not found ... proceeding\n")

        stratified_embeddings_lst = list()
        dim = 0
        for index, row in e_idxlkp.iterrows():
            try:
                if dim is None:
                    dim = len(np.array(entityEmbeddings[int(row[1])]).tolist())
                stratified_embeddings.append(np.array(entityEmbeddings[int(row[1])]).tolist() + [index] )
            except KeyError:
                pass
        embs = pd.DataFrame(stratified_embeddings_lst)
        embs.columns = ["src_" + str(i) for i in range(dim)] + ['label']
        embs.to_csv(rundir+"data\\FB15K\\stratified_embeddings.csv", sep="\t", encoding="UTF-8")



def execute(dim, graph1, graph2):
    prepare_pretrained(CONFIGURATION.rundir, CONFIGURATION.rundir+"w2v.model", CONFIGURATION.src_triples, CONFIGURATION.tgt_triples, dim)
    trainTransE(CONFIGURATION.rundir, dim)
    fill_graph(CONFIGURATION.rundir, graph1, graph2)
