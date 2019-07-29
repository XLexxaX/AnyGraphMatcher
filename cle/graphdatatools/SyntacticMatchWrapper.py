from graphdatatools.GraphToolbox import PipelineDataTuple
import os
import sys
from graphdatatools.InvertedIndexToolbox import InvertedIndex
import pandas as pd
import numpy as np

def interface(main_input, args, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    graph1 = main_input.get(0)
    graph2 = main_input.get(1)
    assert graph1 is not None, "Graph (1) not found in " + os.path.basename(sys.argv[0])
    assert graph2 is not None, "Graph (2) not found in " + os.path.basename(sys.argv[0])
    return exec(graph1, graph2)


def sortkey(val):
    return val[1]

def aggregate_to_dict(indices):
    tmp_tgt_ind = dict()
    for index in indices:
            if index in tmp_tgt_ind.keys():
                tmp_tgt_ind[index] = tmp_tgt_ind[index] + 1
            else:
                tmp_tgt_ind[index] = 1
    return tmp_tgt_ind

def exec(graph1, graph2):
    iindex = InvertedIndex()

    tmp = list()
    for descriptor, resource in graph1.elements.items():
            for relation, literal in resource.literals.items():
                    tmp.append([descriptor] + [literal])
    src = pd.DataFrame(tmp)

    tmp = list()
    for descriptor, resource in graph2.elements.items():
            for relation, literal in resource.literals.items():
                    tmp.append([descriptor] + [literal])
    tgt = pd.DataFrame(tmp)

    iindex = graph2.iindex


    for index, row in src.iterrows():
        indices = iindex.getIndicesForValue(row[1])
        tmp_tgt_ind = aggregate_to_dict(indices)
        best_matching_resources = list(tmp_tgt_ind.items())
        if best_matching_resources is None or best_matching_resources == []:
            continue
        best_matching_resources.sort(key = sortkey, reverse = True)
        best_matching_resources = np.array(best_matching_resources)
        #CONFIGURATION.log(best_matching_resources)

        tmp = tgt.loc[best_matching_resources[:,0]]
        tmp['ngrammatches'] = best_matching_resources[:,1]
        tmp['join'] = 0
        row = row.to_frame().transpose()
        row['join'] = 0
        rows = tmp.merge(row, on='join')
        rows['minlen'] = pd.np.minimum(rows['name_length_x'], rows['name_length_y'])
        rows['minlen'] = rows['minlen']*0.85
        rows = rows.loc[rows['minlen'] <= rows['ngrammatches']]
        for i2, r2 in rows.iterrows():
            graph1.elements[row[0]].synactic_correspondences.append(r2[0])



