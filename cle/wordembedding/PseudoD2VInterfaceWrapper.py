#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import word_tokenize
from configurations.PipelineTools import PipelineDataTuple
import numpy as np
import os
import sys

HETEROGENEITY_THRESHOLD = 0.75
global CONFIGURATION

def interface(main_input, args, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    graph1 = main_input.get(0)
    dim = args.get(0)
    properties = args.get(1)
    assert graph1 is not None, "Graph not found in " + os.path.basename(sys.argv[0])
    assert dim is not None, "Dimension not found in " + os.path.basename(sys.argv[0])
    return execute(graph1, dim, properties)

def execute(graph1, dim, properties):
    predicates, documents = prepare_data(graph1, properties)
    models = train(documents, dim)
    fill_graph(graph1, models, predicates)
    return PipelineDataTuple(graph1)

def fill_graph(graph, models, predicates):
    for descriptor, resource in graph.elements.items():
        for predicate in predicates:
            try:
                vec = np.array(models[predicate].wv[descriptor]).astype(float).tolist()
                resource.embeddings.append(vec)
            except:
                vec = np.array(models[predicate].wv["<>"]).astype(float).tolist()
                resource.embeddings.append(vec)

def array_heterogeneity(x):
    textsset = set()
    textmax = 0
    for elem in x:
        if elem not in textsset:
            textsset.add(elem)
            textmax = textmax + 1
    return textmax/len(x)

def contains_digit(x):
    for elem in x:
        if str(elem).isdigit():
            return True
    return False

def prepare_data(graph, predicates):

    if predicates is None:

        predicates_to_literals = dict()
        # Extract all available literals
        for descriptor, resource in graph.elements.items():
            for predicate, literal in resource.literals.items():
                if graph.elements[predicate] not in predicates_to_literals.keys():
                    predicates_to_literals[predicate] = graph.elements[predicate].texts

        predicates = list()
        # Calculate heterogeneity of literals per predicate and if its higher than a given threshold, use the
        # predicate in the following for constructing documents.
        for descriptor, literals in predicates_to_literals.items():
            if array_heterogeneity(literals) > HETEROGENEITY_THRESHOLD and not contains_digit(literals):
                predicates.append(descriptor)

    # Initialize documents-dict
    documents = dict()
    for predicate in predicates:
        documents[predicate] = list()

    # Create documents by aggregating all literals of 'heterogeneous' predicates as calculated above.

    for descriptor, resource in graph.elements.items():
        for predicate in predicates:
            if predicate in resource.literals.keys():
                documents[predicate].append([descriptor] + resource.literals[predicate].lower().split(" "))
            #else:
                #documents[predicate].append([descriptor, "<>"])

    return predicates, documents


def train(documents, DIMENSIONS):
    if len(documents) < 1:
        return None

    models = dict()
    from wordembedding import EmbeddingHelper
    for predicate, literals in documents.items():
        #literals = [TaggedDocument(doc, [i]) for i, doc in enumerate(literals)]
        models[predicate] = EmbeddingHelper.embed(literals, DIMENSIONS, CONFIGURATION)
        #models[predicate].build_vocab(literals)
        #models[predicate].train(literals, total_examples=models[predicate].corpus_count, epochs=models[predicate].epochs)
        #models[predicate].delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    return models

