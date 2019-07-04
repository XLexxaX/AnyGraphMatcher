#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gensim
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import collections
import smart_open
import random
import re
import numpy as np




MINAVGTEXTLENGTH = 15
MINHETEROGENEITY = 0.75
TEXT_REGEXER = re.compile("\".*\"")
NODEID_REGEXER = re.compile("^<[^<^>]*>")
PROPERTY_REGEXER = re.compile(" <[^<^>]*> ")
TEXTPROPERTIES = list()
nodeid_to_tokenized_docs_index = dict()
doc_vec = list()
textsset = set()
textmax = 0
zerotext_vector = list()


def reset():
    global MINAVGTEXTLENGTH
    global MINHETEROGENEITY
    global TEXT_REGEXER
    global NODEID_REGEXER
    global PROPERTY_REGEXER
    global TEXTPROPERTIES
    global nodeid_to_tokenized_docs_index
    global doc_vec
    global textsset
    global textmax
    global d2v
    d2v=True
    MINAVGTEXTLENGTH = 15
    MINHETEROGENEITY = 0.75
    TEXT_REGEXER = re.compile("\".*\"")
    NODEID_REGEXER = re.compile("^<[^<^>]*>")
    PROPERTY_REGEXER = re.compile(" <[^<^>]*> ")
    TEXTPROPERTIES = list()
    nodeid_to_tokenized_docs_index = dict()
    doc_vec = list()
    textsset = set()
    textmax = 0

def mock_init():
    global d2v
    d2v = False

def embed(triples_path, dim):
    reset()
    init(triples_path, dim)

def init(file, DIMENSIONS):
    global nodeid_to_tokenized_docs_index
    tokenized_docs, nodeid_to_tokenized_docs_index = prepare_data(file)
    print("Training document embeddings for "+str(file)+"...")
    model = train(tokenized_docs, DIMENSIONS)
    generate_embeddings(tokenized_docs, model)
    print("Learned document embeddings for "+str(file)+".\n")


def np_arraylength_aggregator(x):
    return len(x)

def np_arrayheterogeneity_aggregator(x):
    global textsset
    global textmax
    if x not in textsset:
        textsset.add(x)
        textmax = textmax + 1
    return textmax


def fill_text_properties(file):
    global TEXTPROPERTIES
    global MINAVGTEXTLENGTH
    global MINHETEROGENEITY
    global textsset
    global textmax
    properties_dict = dict()
    for line in open(file):
        try:
            text = TEXT_REGEXER.findall(line)[0].replace("\"", "")
            prop = PROPERTY_REGEXER.findall(line)[0].replace(" ", "")
            if (prop in properties_dict):
                properties_dict[prop] = properties_dict[prop] + [text]
            else:
                properties_dict[prop] = [text];
        except:
            pass


    for property, texts in properties_dict.items():
        textmax = 0
        texts = np.extract((not (texts == '')), texts)
        nptexts = np.atleast_1d(np.array(list(texts)))
        func1 = np.vectorize(np_arraylength_aggregator)  # or use a different name if you want to keep the original f
        func2 = np.vectorize(np_arrayheterogeneity_aggregator)

        textsset = set()
        textmax = 0
        if (np.sum(func1(nptexts))/len(nptexts) > MINAVGTEXTLENGTH and np.max(func2(nptexts))/len(nptexts) > MINHETEROGENEITY):
            TEXTPROPERTIES = TEXTPROPERTIES + [property]


def prepare_data(file):
    fill_text_properties(file)
    docs = dict()
    for line in open(file):
        try:
            text = TEXT_REGEXER.findall(line)[0].replace("\"", "")
            nodeid = NODEID_REGEXER.findall(line)[0].replace("\"", "")
            prop = PROPERTY_REGEXER.findall(line)[0].replace(" ", "")
            if (nodeid in docs and prop in TEXTPROPERTIES):
                tmp = docs[nodeid]
                tmp[prop] = text
                docs[nodeid] = tmp
            elif (prop in TEXTPROPERTIES):
                docs[nodeid] = dict({prop: text})
        except:
            pass

    tokenized_docs = list()
    nodeid_to_tokenized_docs_index = dict()
    tokenized_docs_index_to_property = dict()
    ctr = 0
    for nodeid, properties_and_corpuses in docs.items():
        for property, corpus in properties_and_corpuses.items():
            if (nodeid in nodeid_to_tokenized_docs_index):
                tmp = nodeid_to_tokenized_docs_index[nodeid]
                tmp[property] = ctr
                nodeid_to_tokenized_docs_index[nodeid] = tmp
            else:
                tmp = dict()
                tmp[property] = ctr
                nodeid_to_tokenized_docs_index[nodeid] = tmp
            tokenized_docs = tokenized_docs + [corpus.split()]
            ctr = ctr + 1

    return tokenized_docs, nodeid_to_tokenized_docs_index


def train(tokenized_docs, DIMENSIONS):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized_docs)]
    model = Doc2Vec(vector_size=DIMENSIONS, min_count=2, epochs=40)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    return model

# In[4]:

def generate_embeddings(tokenized_docs, model):
    global doc_vec
    global zerotext_vector
    for tokenized_corpus in tokenized_docs:
        doc_vec = doc_vec + [model.infer_vector(tokenized_corpus)]
    zerotext_vector = model.infer_vector([""])

# In[6]:



def get_embedding(resource):
    global d2v
    if not d2v:
        return []

    #resource is e.g. "<http://rdata2graph.sap.com/Amazon1/resource/1ea15173-3c1c-425c-9dbb-8f1a785aff5e>"
    global doc_vec
    global TEXTPROPERTIES
    global DIMENSIONS
    global nodeid_to_tokenized_docs_index
    #return [list(doc_vec[index]) for prop, index in [prop_to_index for nodeid, prop_to_index in nodeid_to_tokenized_docs_index[resource].items()]]
    vector = []
    for p in TEXTPROPERTIES:
        try:
            vector = vector + list(doc_vec[nodeid_to_tokenized_docs_index[resource][p]])
        except:
            vector = vector + list(zerotext_vector)
    return vector

