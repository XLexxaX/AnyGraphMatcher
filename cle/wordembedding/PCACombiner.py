import pandas as pd
#from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from cle.configurations.PipelineTools import PipelineDataTuple
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
    assert graph1 is not None, "Graph not found in " + os.path.basename(sys.argv[0])
    assert dim is not None, "Dimension not found in " + os.path.basename(sys.argv[0])
    if graph2 is None:
        return execute(graph1)
    else:
        return PipelineDataTuple(execute(graph1).elems[0], execute(graph2).elems[0])

def execute(graph, dim=20):
    embeddings = None
    for descriptor, resource in graph.elements.items():
        if embeddings is None:
            embeddings = np.array([[descriptor] + resource.embeddings[0]])
        else:
            embeddings = np.append(embeddings, [[descriptor] + resource.embeddings[0]], axis=0)
    df = pd.DataFrame(embeddings)
    df.set_index(0)

    pca = decomposition.KernelPCA(n_components=dim, kernel='rbf')
    reduced_df = pca.fit_transform(df[[df.columns[i] for i in range(len(df.columns)) if not i == 0]])
    reduced_df = pd.DataFrame(reduced_df)
    reduced_df.loc[:,dim] = df[0]
    reduced_df = reduced_df.set_index(dim)
    for descriptor, reduced_embedding in reduced_df.iterrows():
        graph.elements[descriptor].embeddings[0] = reduced_embedding.tolist()

    return PipelineDataTuple(graph)
