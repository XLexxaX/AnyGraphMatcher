

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from cle.configurations.PipelineTools import PipelineDataTuple
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
            #print("Key " + descriptor + " not found ... proceeding")

def array_heterogeneity(x):
    textsset = set()
    textmax = 0
    for elem in x:
        if elem not in textsset:
            textsset.add(elem)
            textmax = textmax + 1
    return textmax/len(x)

def prepare_data(graph):
    documents = list()
    for descriptor, resource in graph.elements.items():
        for predicate, literal in resource.literals.items():
            documents.append([descriptor, predicate, literal])
        for predicate, relation in resource.relations.items():
            documents.append([descriptor, predicate, relation.descriptor])

    return documents


def train(documents, DIMENSIONS):
    if len(documents) < 1:
        return None
    from cle.wordembedding import EmbeddingHelper
    model = EmbeddingHelper.embed(documents, DIMENSIONS, CONFIGURATION)
    return model
