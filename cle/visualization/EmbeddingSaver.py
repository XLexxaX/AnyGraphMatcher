import numpy as np
import pandas as pd
import sys
import os
from graphdatatools.GraphToolbox import PipelineDataTuple

CONFIGURATION = None

def interface(main_input, args, conf):
    global CONFIGURATION
    CONFIGURATION = conf
    graph1 = main_input.get(0)
    graph2 = main_input.get(1)
    assert graph1 is not None, "Graph (1) not found in " + os.path.basename(sys.argv[0])
    assert graph2 is not None, "Graph (2) not found in " + os.path.basename(sys.argv[0])
    execute(graph1, graph2)
    return PipelineDataTuple(graph1, graph2)


def execute(graph1, graph2):


    # Reduce dimensionality
    vecs = list()
    #ids1 = np.random.choice(np.array(list(graph1.elements.keys())), size=min(50000,len(graph1.elements.keys())))
    #ids2 = np.random.choice(np.array(list(graph2.elements.keys())), size=min(50000,len(graph2.elements.keys())))
    ids1 = np.array(list(graph1.elements.keys()))
    ids2 = np.array(list(graph2.elements.keys()))
    ids = np.concatenate((ids1, ids2), axis=0)


    output = list()
    vlen = len(graph1.elements[ids[0]].embeddings[0])
    out = open(CONFIGURATION.rundir + 'stratified_embeddings.csv', mode="w+", encoding="UTF-8")
    columns = ['src_' + str(i) for i in range(0, vlen)] + ['label']#, 'category', 'origin']
    out.write(str("\t".join(columns) + "\n"))
    # Plot embeddings
    for i, label in enumerate(ids):

            try:
                v = graph1.elements[label].embeddings[0]
            except:
                v = graph2.elements[label].embeddings[0]

            if vlen==0:
                vlen = len(v)
            if i < len(ids1):
                try:
                    cat = graph1.elements[label].relations[
                        'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'].descriptor
                except KeyError:
                    cat = 'none'
                graph = "graph1"
                output.append(v+[label,cat,'graph1'])
            else:
                try:
                    cat = graph2.elements[label].relations[
                        'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'].descriptor
                except KeyError:
                    cat = 'none'
                graph = "graph2"
                #output.append(v+[label,cat,'graph2'])
            #line = v+[label,cat,'graph2']
            line=v+[label]
            line = [str(val) for val in line]
            out.write(str("\t".join(line) + "\n"))
    out.close()
    #pd.DataFrame(np.array(output), columns=['x'+str(i) for i in range(0,vlen)]+['label','category','origin']).to_csv(path_or_buf=CONFIGURATION.rundir+'stratified_embeddings.csv')



