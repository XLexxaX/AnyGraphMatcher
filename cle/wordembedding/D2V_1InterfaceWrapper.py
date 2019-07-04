#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from cle.configurations.PipelineTools import PipelineDataTuple
import numpy as np
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
    properties = args.get(1)
    assert graph1 is not None, "Graph (1) not found in " + os.path.basename(sys.argv[0])
    assert graph2 is not None, "Graph (2) not found in " + os.path.basename(sys.argv[0])
    assert dim is not None, "Dimension not found in " + os.path.basename(sys.argv[0])
    return execute(graph1, graph2, dim, properties)

def execute(graph1, graph2, dim, properties):
    predicates1, documents1 = prepare_data(graph1, properties)
    predicates2, documents2 = prepare_data(graph2, properties)
    documents = documents1 + documents2
    model = train(documents, dim)
    fill_graph(graph1, model, predicates1)
    fill_graph(graph2, model, predicates2)
    return PipelineDataTuple(graph1, graph2)

def fill_graph(graph, model, predicates):
    for descriptor, resource in graph.elements.items():
        tmp = ""
        for predicate, literal in resource.literals.items():
            if predicate in predicates:
                tmp = tmp + " " + literal
        doc = word_tokenize(tmp.lower())
        if tmp is None or tmp=="":
           doc = [""]
        test = np.array(model.infer_vector(doc)).astype(float).tolist()
        resource.embeddings.append(test)

def array_heterogeneity(x):
    textsset = set()
    textmax = 0
    for elem in x:
        if elem not in textsset:
            textsset.add(elem)
            textmax = textmax + 1
    return textmax/len(x)

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
            if array_heterogeneity(literals) > HETEROGENEITY_THRESHOLD:
                predicates.append(descriptor)

    documents = list()
    # Create documents by aggregating all literals of 'heterogeneous' predicates as calculated above.
    for descriptor, resource in graph.elements.items():
        tmp = []
        for predicate, literal in resource.literals.items():
            if predicate in predicates:
                tmp = tmp + [[literal]]
        # Finally, up-sample small strings according to the largest string.
        maxdoclen = 0
        for t in tmp:
            if len(t) > maxdoclen:
                maxdoclen = len(t)
        for index, value in enumerate(tmp):
            i = 0
            while len(value) < maxdoclen:
                value = value + [value[i]]
                tmp[index] = value
                i = i + 1
        documents.append(word_tokenize(" ".join([item for sublist in tmp for item in sublist]).lower()))

    return predicates, documents


def train(documents, DIMENSIONS):
    if len(documents) < 1:
        return None
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
    model = Doc2Vec(vector_size=DIMENSIONS, min_count=1, epochs=40)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    return model
