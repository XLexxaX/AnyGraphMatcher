from wordembedding.word2vec import word2vec_embedding_from_sentences_v2
from configurations.PipelineTools import PipelineDataTuple
import numpy as np
import os
import sys
global CONFIGURATION

def interface(main_input, args, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    graph1 = main_input.get(0)
    graph2 = main_input.get(1)
    dim = args.get(0)
    assert graph1 is not None, "Graph (1) not found in " + os.path.basename(sys.argv[0])
    assert graph2 is not None, "Graph (2) not found in " + os.path.basename(sys.argv[0])
    assert dim is not None, "Dimension not found in " + os.path.basename(sys.argv[0])
    return execute(graph1, graph2, dim)

def execute(graph1, graph2, dim):
    if graph1.corpus is None or graph2.corpus is None:
        print("!!! At least one of the graphs has no corpus !!!")
        return PipelineDataTuple(graph1, graph2)
    full_corpus = graph1.corpus + graph2.corpus
    model = word2vec_embedding_from_sentences_v2(full_corpus, CONFIGURATION, sg=0, size=dim, window=500)
    for descriptor, resource in graph1.elements.items():
        try:
            resource.embeddings.append(np.array(model[descriptor.lower()]).astype(float).tolist())
        except KeyError:
            resource.embeddings.append(np.array(model["<>"]).astype(float).tolist())
            print("Key " + descriptor + " not found ... proceeding")
    for descriptor, resource in graph2.elements.items():
        try:
            resource.embeddings.append(np.array(model[descriptor.lower()]).astype(float).tolist())
        except KeyError:
            resource.embeddings.append(np.array(model["<>"]).astype(float).tolist())
            print("Key " + descriptor + " not found ... proceeding")
    return PipelineDataTuple(graph1, graph2)
