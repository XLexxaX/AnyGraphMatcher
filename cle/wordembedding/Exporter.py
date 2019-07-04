import sys
import os
import uuid
import numpy as np
from sklearn.manifold import TSNE
from cle.graphdatatools.GraphToolbox import PipelineDataTuple
import json
import codecs
global CONFIGURATION

def interface(main_input, args, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    graph1 = main_input.get(0)
    graph2 = main_input.get(1)
    assert graph1 is not None, "Graph not found in " + os.path.basename(sys.argv[0])
    a = execute(graph1)
    if graph2 is not None:
        b = execute(graph2)
        return (PipelineDataTuple(a.elems[0], b.elems[0]))
    else:
        return a

def execute(graph1):
    X = None
    Y = None

    for descriptor, resource in graph1.elements.items():

        if X is None:
            X = [resource.embeddings[0]]
            try:
                type = resource.relations['http://www.w3.org/1999/02/22-rdf-syntax-ns#type'].descriptor
            except KeyError:
                type = "nd"
            Y = [[descriptor, resource.type.to_string(), type]]
        else:
            X = X + [resource.embeddings[0]]
            try:
                type = resource.relations['http://www.w3.org/1999/02/22-rdf-syntax-ns#type'].descriptor
            except KeyError:
                type = "nd"
            Y = Y + [[descriptor, resource.type.to_string(), type]]
    X = np.array(X)
    Y = np.array(Y)
    #X2 = TSNE(n_components=2).fit_transform(X)
    X2 = np.concatenate((X, Y), axis=1)
    X2 = X2.tolist()

    filepath = "../"+str(uuid.uuid4())+".csv"
    with codecs.open(filepath, 'w+', encoding='utf8') as f:
        for row in X2:
            f.write(str(row).replace("http://rdata2graph.sap.com/hilti","").replace("'","").replace(" ","").replace("[","").replace("]","")+"\n")
    return PipelineDataTuple(graph1)

