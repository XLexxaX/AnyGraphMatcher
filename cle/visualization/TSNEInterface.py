import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
from matplotlib.lines import Line2D

from graphdatatools.GraphToolbox import PipelineDataTuple
CONFIGURATION = None

def interface(main_input, args, conf):
    global CONFIGURATION
    CONFIGURATION = conf
    graph1 = main_input.get(0)
    graph2 = main_input.get(1)
    dim = args.get(0)
    assert graph1 is not None, "Graph (1) not found in " + os.path.basename(sys.argv[0])
    #assert graph2 is not None, "Graph (2) not found in " + os.path.basename(sys.argv[0])
    assert dim is not None, "No dimension given in " + os.path.basename(sys.argv[0])
    execute(graph1, graph2, dim)
    if graph2 is not None:
        return PipelineDataTuple(graph1, graph2)
    else:
        return PipelineDataTuple(graph1)


def execute(graph1, graph2, dim):

    # Reduce dimensionality
    vecs = list()
    ids1 = np.array(list(graph1.elements.keys()))
    if graph2 is not None:
        ids2 = np.array(list(graph2.elements.keys()))
        ids = np.concatenate((ids1, ids2), axis=0)
    else:
        ids = ids1

    for descriptor in ids:
        try:
            vecs.append(graph1.elements[descriptor].embeddings[0])
        except:
            vecs.append(graph2.elements[descriptor].embeddings[0])

    vecs = np.array(vecs)
    assert len(vecs.shape) > 1, 'Resources have different embeddings sizes.'
    if vecs.shape[1] > dim:
        from sklearn.manifold import TSNE
        vecs = TSNE(n_components=dim).fit_transform(vecs)
    else:
        return


    for i, label in enumerate(ids):
        try:
            graph1.elements[label].embeddings[0] = vecs[i]
        except:
            graph2.elements[label].embeddings[0] = vecs[i]





