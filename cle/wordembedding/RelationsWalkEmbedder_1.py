#!/usr/bin/env python
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
    sentence_generation_method = args.get(1)
    ngrams = args.get(2)
    maxdepth = args.get(3)
    assert graph1 is not None, "Graph1 not found in " + os.path.basename(sys.argv[0])
    assert graph2 is not None, "Graph2 not found in " + os.path.basename(sys.argv[0])
    assert dim is not None, "Dimension not found in " + os.path.basename(sys.argv[0])
    assert sentence_generation_method is not None, "Sentence generation method not found in " + os.path.basename(sys.argv[0])
    return execute(graph1, graph2, dim, sentence_generation_method, ngrams, maxdepth)

def execute(graph1, graph2, dim, sentence_generation_method, ngrams=False, maxdepth=1):
    documents = prepare_data(graph1, sentence_generation_method, ngrams, maxdepth)
    documents = documents + prepare_data(graph2, sentence_generation_method, ngrams, maxdepth)
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

def prepare_data(graph, sentence_generation_method, ngrams, maxdepth):
    documents = list()
    maxdepth = [maxdepth]
    total_ctr = len(maxdepth)*len(graph.elements.keys())
    ctr = 0
    for i in maxdepth:
        for descriptor, resource in graph.elements.items():
            if sentence_generation_method == 'steps':
                tmp = deep_steps(resource, 0, i, graph, "", ngrams)
            elif sentence_generation_method == 'batch':
                tmp = broad_step(resource, 0, i, graph, "", ngrams)
            documents = documents + tmp #.append(tmp)
            ctr +=1
            print("      --> Generating training corpus: " + str(int(100*ctr/total_ctr)) + "%", end="\r")
    print("      --> Generating training corpus: 100%")
    return documents

def deep_steps(resource, i, maxdepth, graph, exclude_descriptor, ngrams=False):
    if i>maxdepth:
        return None
    new_sentences = list()
    for predicate, literal in resource.literals.items():
        if ngrams:

            new_sentences.append([predicate])
        else:
            new_sentences.append([predicate])
    for predicate, relation in resource.relations.items():
        if not relation == exclude_descriptor:
            tmp = deep_steps(relation, i+1, maxdepth, graph, resource.descriptor)
            if tmp is not None:
                for sentence in tmp:
                    new_sentences.append([predicate] + sentence)
    new_sentences2 = list()
    for sentence in new_sentences:
        new_sentences2.append([resource.descriptor] + sentence)

    return new_sentences2

def broad_step(resource, i, maxdepth, graph, exclude_descriptor, ngrams=False):
    sentence = [resource.descriptor]
    if i>maxdepth:
        return sentence
    for predicate, literal in resource.literals.items():
        if ngrams:
            sentence = sentence + [predicate]
        else:
            sentence = sentence + [predicate]
    for predicate, relation in resource.relations.items():
        if not relation == exclude_descriptor:
            sentence = sentence + [predicate]
            sentence = sentence + broad_step(relation, i+1, maxdepth, graph, resource.descriptor)

    #for descriptor, res in graph.elements.items():
    #    for predicate, relation in resource.relations.items():
    #        if relation.descriptor == resource.descriptor:
    #            #sentence = sentence + [descriptor]
    #            sentence = sentence + broad_step(relation, i+1, maxdepth, graph, resource.descriptor)

    return sentence

def train(documents, DIMENSIONS):
    if len(documents) < 1:
        return None
    from wordembedding import EmbeddingHelper
    model = EmbeddingHelper.embed(documents, DIMENSIONS, CONFIGURATION)
    return model


