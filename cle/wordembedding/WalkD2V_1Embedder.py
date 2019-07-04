#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
    documents, documents_ids = prepare_data(graph1, dict(), list())
    documents, documents_ids = prepare_data(graph2, documents_ids, documents)

    global ctr
    documents.append(["<>","<>"])
    documents_ids["<>"] = ctr
    ctr += 1

    model = train(documents, dim)
    with open(CONFIGURATION.rundir + "document_ids.csv", mode="w+") as f:
        for descriptor, index in documents_ids.items():
            f.write(descriptor + "," + str(index) + "\n")
    fill_graph(graph1, model, documents_ids)
    fill_graph(graph2, model, documents_ids)
    return PipelineDataTuple(graph1, graph2)

def fill_graph(graph, model, documents_ids):
    for descriptor, resource in graph.elements.items():
        try:
            test = np.array(model.docvecs[documents_ids[descriptor]]).astype(float).tolist()
            resource.embeddings.append(test)
        except KeyError:
            test = np.array(model.docvecs[documents_ids["<>"]]).astype(float).tolist()
            resource.embeddings.append(test)
            #print("Key " + descriptor + " not found ... proceeding")

def array_heterogeneity(x):
    textsset = set()
    textmax = 0
    for elem in x:
        if elem not in textsset:
            textsset.add(elem)
            textmax = textmax + 1
    return textmax/len(x)

global ctr
ctr = 0
def prepare_data(graph, documents_ids, aggregated_documents):
    global ctr
    documents = dict()
    for i in [0]:
        for descriptor, resource in graph.elements.items():
            tmp = broad_step(resource, 0, i, graph, "")
            #for sentence in broad_step(resource, 0, i, graph, ""):
            if descriptor in documents.keys():
                documents[descriptor] = documents[descriptor] + tmp
            elif not " ".join(tmp) == '' and len(" ".join(tmp))>1:
                documents_ids[descriptor] = ctr
                ctr += 1
                documents[descriptor] = tmp

            #if descriptor in documents.keys():
            #    documents[descriptor] = documents[descriptor] + tmp
            #elif not len(tmp)<1:
            #    documents_ids[descriptor] = ctr
            #    ctr += 1
            #    documents[descriptor] = tmp
    documents['<>'] = ["<>","<>","<>","<>","<>"]
    documents_ids['<>'] = ctr
    for descriptor, sentences in documents.items():
            aggregated_documents.append(sentences)
    return aggregated_documents, documents_ids

import re
def deep_steps(resource, i, maxdepth, graph, exclude_descriptor):
    if i>maxdepth:
        return []
    new_sentences = list()
    for predicate, literal in resource.literals.items():
        if not literal == '':
            new_sentences.append([literal])
    for predicate, relation in resource.relations.items():
        if not relation == exclude_descriptor:
            tmp = deep_steps(relation, i+1, maxdepth, graph, resource.descriptor)
            if not tmp == []:
                for sentence in tmp:
                    new_sentences.append([predicate, relation.descriptor])

    return new_sentences

def broad_step(resource, i, maxdepth, graph, exclude_descriptor):
    sentence = []
    #sentence = []
    if i>maxdepth:
        return sentence
    for predicate, literal in resource.literals.items():
        literal = re.sub('[^A-z0-9<> ]','', literal.lower())
        if not literal == '' and not literal == ' ':
            if len(literal)>3:
                sentence = sentence + [literal]
            else:
                sentence = sentence + [literal]
    for predicate, relation in resource.relations.items():
        if not relation == exclude_descriptor:
            sentence = sentence + broad_step(relation, i+1, maxdepth, graph, resource.descriptor)

    #for descriptor, res in graph.elements.items():
    #    for predicate, relation in resource.relations.items():
    #        if relation.descriptor == resource.descriptor:
    #            sentence = sentence + broad_step(relation, i+1, maxdepth, graph, resource.descriptor)

    return sentence

def train(documents, DIMENSIONS):
    if len(documents) < 1:
        return None
    from cle.wordembedding import D2VEmbeddingHelper
    model = D2VEmbeddingHelper.embed(documents, DIMENSIONS, CONFIGURATION)
    return model


