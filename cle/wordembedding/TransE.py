#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#from nltk.tokenize import word_tokenize
from configurations.PipelineTools import PipelineDataTuple
import numpy as np
from wordembedding.TransE_modules.prepare_pretrained import load as prepare_pretrained
from wordembedding.TransE_modules.Train import trainTransE as trainTransE

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
    execute(dim)
    return PipelineDataTuple(graph1, graph2)

def execute(dim):
    transe_dir = "wordembedding/TransE/data/FB15K/"
    prepare_pretrained(CONFIGURATION.rundir, CONFIGURATION.rundir+"w2v.model", CONFIGURATION.src_triples, CONFIGURATION.tgt_triples, transe_dir, dim)
    trainTransE(CONFIGURATION.rundir, dim)
