#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from configurations.PipelineTools import PipelineDataTuple
import numpy as np
from gensim.models import Word2Vec
import pandas as pd

import os
import sys

HETEROGENEITY_THRESHOLD = 0.75
global CONFIGURATION
global predicates
predicates = list()
global predicates_thresh
predicates_thresh = 0

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
    save_data(graph1, graph2, model)
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


def save_data(graph1, graph2, model):

    global predicates_thresh
    global predicates
    ids = np.array(predicates)
    vecs = list()
    for descriptor in ids:
            vecs.append(model.wv[descriptor[1]])
    vecs = np.array(vecs)

    output = list()
    # Plot embeddings
    for i, label in enumerate(ids):
            x, y = vecs[i,0], vecs[i,1]

            if i < predicates_thresh:
                output.append([x,y,label[1], label[0],'graph1'])
            else:
                output.append([x,y,label[1], label[0],'graph2'])
    pd.DataFrame(np.array(output), columns=['x','y','label','original_label','origin']).to_csv(path_or_buf=CONFIGURATION.rundir+'predicate_embeddings.csv')


def prepare_data(graph):
    documents = list()
    ctr=0
    global predicates
    for descriptor, resource in graph.elements.items():
        #tmp = list()
        #literal_predicates=list()
        for predicate, literal in resource.literals.items():
            #tmp = tmp  + [descriptor, predicate, literal]
            #x = x+[predicate, literal]
            predicates.append([predicate, predicate + str(ctr)])
            predicate = predicate + str(ctr)
            ctr += 1
            documents.append([resource.descriptor, predicate, literal])
            #literal_predicates.append(predicate)
            #documents.append([descriptor, literal])
        for predicate, relation in resource.relations.items():
            predicates.append([predicate, predicate + str(ctr)])
            predicate = predicate + str(ctr)
            if 'prdha' in predicate:
                CONFIGURATION.log(predicate)
            ctr += 1
            documents.append([descriptor, predicate, relation.descriptor])
            #tmp = tmp  + [descriptor, predicate, relation.descriptor]
            #x=x+[predicate, relation.descriptor]
            #documents.append(literal_predicates + [predicate])
        #documents.append(tmp)
    global predicates_thresh
    if predicates_thresh == 0:
         predicates_thresh = ctr

    return documents


def train(documents, DIMENSIONS):
    if len(documents) < 1:
        return None
    from wordembedding import EmbeddingHelper
    model = EmbeddingHelper.embed(documents, DIMENSIONS, CONFIGURATION)


    import pandas as pd
    # Reduce dimensionality
    vecs = list()
    #ids1 = np.random.choice(np.array(list(graph1.elements.keys())), size=min(10000,len(graph1.elements.keys())))
    #ids2 = np.random.choice(np.array(list(graph2.elements.keys())), size=min(10000,len(graph2.elements.keys())))
    #ids1 = np.array(list(graph1.elements.keys()))
    #ids2 = np.array(list(graph2.elements.keys()))
    ids1=list()
    ids2=list()





    return model


