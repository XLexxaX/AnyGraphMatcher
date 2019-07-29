#/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from configurations.PipelineTools import PipelineDataTuple
import numpy as np
from gensim.models import Word2Vec

import os
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
    return execute(graph1, graph2, dim)

def execute(graph1, graph2, dim):
    documents = prepare_data(graph1)
    documents = documents + prepare_data(graph2)
    model = train(documents, dim)
    fill_graph(graph1, model)
    fill_graph(graph2, model)
    return PipelineDataTuple(graph1, graph2)

def fill_graph(graph, model):
    for descriptor, resource in graph.elements.items():
        try:
            test = np.array(model.wv[descriptor]).astype(float).tolist()
            resource.embeddings.append(test)
        except KeyError:
            resource.embeddings.append(model.wv['<>'].astype(float).tolist())
            #CONFIGURATION.log("Key " + descriptor + " not found ... proceeding")

def array_heterogeneity(x):
    textsset = set()
    textmax = 0
    for elem in x:
        if elem not in textsset:
            textsset.add(elem)
            textmax = textmax + 1
    return textmax/len(x)

def prepare_data(graph):
    corr = dict()
    corr_values = dict()
    with open("C:/Users/D072202/DeepAnyMatch/DeepAnyMatch/data/sap_hilti_data/balanced_walks/final_trainset.csv") as f:
        for line in f:
            line = line.split("\t")
            if str(line[2]) == "1":
                corr[0] = corr[1]
                corr[1] = corr[0]
    documents = list()
    for descriptor, resource in graph.elements.items():
        #tmp = list()
        x=list()
        for predicate, literal in resource.literals.items():
            #tmp = tmp  + [descriptor, predicate, literal]
            x = x+[predicate, literal]
        for predicate, relation in resource.relations.items():
            #tmp = tmp  + [descriptor, predicate, relation.descriptor]
            x=x+[predicate, relation.descriptor]
        x=x+[descriptor]
        if descriptor in corr.keys():
            corr_values[descriptor] = x
        #documents.append(tmp)

    for descriptor, resource in graph.elements.items():
        #tmp = list()
        x=list()
        for predicate, literal in resource.literals.items():
            #tmp = tmp  + [descriptor, predicate, literal]
            x = x+[predicate, literal]
        for predicate, relation in resource.relations.items():
            #tmp = tmp  + [descriptor, predicate, relation.descriptor]
            x=x+[predicate, relation.descriptor]
            if relation.descriptor in corr.keys():
                x = x + corr_values[corr[relation.descriptor]]
        x=x+[descriptor]
        if descriptor in corr.keys():
            x = x + corr_values[corr[descriptor]]
        #documents.append(tmp)
        documents.append(x)

    return documents


def train(documents, DIMENSIONS):
    if len(documents) < 1:
        return None
    from wordembedding import EmbeddingHelper
    model = EmbeddingHelper.embed(documents, DIMENSIONS, CONFIGURATION)
    return model

