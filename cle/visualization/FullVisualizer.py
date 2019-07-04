import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
from matplotlib.lines import Line2D

from cle.graphdatatools.GraphToolbox import PipelineDataTuple
CONFIGURATION = None

def interface(main_input, args, conf):
    global CONFIGURATION
    CONFIGURATION = conf
    graph1 = main_input.get(0)
    graph2 = main_input.get(1)
    assert graph1 is not None, "Graph (1) not found in " + os.path.basename(sys.argv[0])
    assert graph2 is not None, "Graph (2) not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.gold_mapping is not None, "Path to gold mapping not found in " + os.path.basename(sys.argv[0])
    execute(graph1, graph2)
    return PipelineDataTuple(graph1, graph2)


def execute(graph1, graph2):
    draw_embeddings(graph1, graph2)


def draw_embeddings(graph1, graph2):
    # Reduce dimensionality
    vecs = list()
    ids1 = np.random.choice(np.array(list(graph1.elements.keys())), size=min(10000,len(graph1.elements.keys())))
    ids2 = np.random.choice(np.array(list(graph2.elements.keys())), size=min(10000,len(graph2.elements.keys())))
    ids = np.concatenate((ids1, ids2), axis=0)

    for descriptor in ids:
        try:
            vecs.append(graph1.elements[descriptor].embeddings[0])
        except:
            vecs.append(graph2.elements[descriptor].embeddings[0])
    vecs = np.array(vecs)
    #if vecs.shape[1] > 2:
    #    from sklearn.manifold import TSNE
    #    vecs = TSNE(n_components=2).fit_transform(vecs)
    assert vecs.shape[1] == 2, "Visualization mechanism can only work with 2-dimensional embeddings."


    # Plot embeddings
    plt.figure(figsize=(40,40))
    for i, label in enumerate(ids):
            x, y = vecs[i,0], vecs[i,1]

            if i < len(ids1):
                plt.scatter(x, y, color='blue', s=500)
            else:
                plt.scatter(x, y, color='red', s=500)
    plt.legend([Line2D([0], [0], color='blue', lw=4), Line2D([0], [0], color='red', lw=4)], ['Graph1', 'Graph2'])
    plt.savefig(CONFIGURATION.rundir+"embeddings_full.png")
    plt.close()

