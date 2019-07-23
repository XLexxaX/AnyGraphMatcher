import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from graphdatatools.GraphToolbox import PipelineDataTuple
CONFIGURATION = None

def interface(main_input, args, conf):
    global CONFIGURATION
    CONFIGURATION = conf
    graph1 = main_input.get(0)
    graph2 = main_input.get(1)
    assert graph1 is not None, "Graph (1) not found in " + os.path.basename(sys.argv[0])
    assert graph2 is not None, "Graph (2) not found in " + os.path.basename(sys.argv[0])
    CONFIGURATION.log("Graph1:\n")
    execute(graph1)
    CONFIGURATION.log("Graph2:\n")
    execute(graph2)
    return PipelineDataTuple(graph1, graph2)

def execute(graph):
    emb_size = len(graph.elements[list(graph.elements.keys())[0]].embeddings[0])
    att_matrix = pd.DataFrame([], columns=['name']+['axis'+str(i) for i in list(range(emb_size))])
    ids = np.random.choice(np.array(list(graph.elements.keys())), size=min(250,len(graph.elements)))
    for descriptor in ids:
        values = [descriptor]
        values = values + graph.elements[descriptor].embeddings[0]
        att_matrix = att_matrix.append(pd.DataFrame([values], columns=att_matrix.columns))
    att_matrix = att_matrix.reset_index(drop=True)

    training_material = pd.read_csv(CONFIGURATION.rundir + 'w2v_formatted_training_material.csv', header=None, index_col=False)


    for axis in ['axis'+str(i) for i in list(range(emb_size))]:
        most_frequent_common_value = list()
        axis_values = list()
        clustering = KMeans(n_clusters=3).fit(pd.DataFrame(att_matrix[axis]))
        np.array(clustering.labels_)
        for label in set(clustering.labels_):
            tmp = att_matrix.loc[np.where(clustering.labels_==label)]
            tmp2 = training_material.loc[(training_material[0].isin(list(tmp.name))) | (training_material[1].isin(list(tmp.name)))]
            x = [x for x in list(tmp2[0]) + list(tmp2[1]) if x not in list(tmp.name)]
            most_frequent_common_value.append(max(set(x), key = x.count))
            axis_values.append(x)
        x = None
        for vals in axis_values:
            tmp = set()
            for val in vals:
                if axis_values.count(val) > 1:
                    tmp.add(val)
            if x is not None:
                x = x.symmetric_difference(tmp)
            else:
                x = tmp
        CONFIGURATION.log('\n')
        CONFIGURATION.log('This is '+axis+'; the most frequently used co-occurences for the embedding-training in the clusters were:')
        CONFIGURATION.log(str(most_frequent_common_value))
        CONFIGURATION.log('This is '+axis+'; the set difference in the embedding-training among the clusters were:')
        CONFIGURATION.log(list(x))
        CONFIGURATION.log('-------------\n\n')

