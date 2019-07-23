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
    assert graph1 is not None, "Graph (1) not found in " + os.path.basename(sys.argv[0])
    assert graph2 is not None, "Graph (2) not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.gold_mapping is not None, "Path to gold mapping not found in " + os.path.basename(sys.argv[0])
    execute(graph1, graph2)
    return PipelineDataTuple(graph1, graph2)


def execute(graph1, graph2):
    draw_embeddings(graph1, graph2)
    draw_alignments(graph1, graph2)


def draw_embeddings(graph1, graph2):


    # Reduce dimensionality
    vecs = list()
    ids1 = np.random.choice(np.array(list(graph1.elements.keys())), size=min(60,len(graph1.elements.keys())))
    ids2 = np.random.choice(np.array(list(graph2.elements.keys())), size=min(60,len(graph2.elements.keys())))
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
                plt.scatter(x, y, color='blue',s=500)
                plt.annotate(label, xy=(x, y), xytext=(5, 2),
                    textcoords='offset points', ha='right', va='bottom', color='blue')
            else:
                plt.scatter(x, y, color='red',s=500)
                plt.annotate(label, xy=(x, y), xytext=(5, 2),
                    textcoords='offset points', ha='right', va='bottom', color='red')
    plt.legend([Line2D([0], [0], color='blue', lw=4), Line2D([0], [0], color='red', lw=4)], ['Graph1', 'Graph2'])
    plt.savefig(CONFIGURATION.rundir+"embeddings.png")
    plt.close()

def draw_alignments(graph1, graph2):

    #gold_mapping = pd.read_csv(CONFIGURATION.gold_mapping, delimiter="\t", header=None, skiprows=1)
    #gold_mapping = gold_mapping.sample(n=min(20, len(gold_mapping)))
    #gold_mapping = gold_mapping.applymap(lambda s: s.lower() if type(s) == str else s)
    #gold_mapping.columns = ["gold_src_id", "gold_tgt_id", "label"]
#
    #vecs = list()
    #for index, row in gold_mapping.iterrows():
    #    vecs.append(graph1.elements[row['gold_src_id']].embeddings[0])
    #    vecs.append(graph2.elements[row['gold_tgt_id']].embeddings[0])
    #vecs = np.array(vecs)
    #if vecs.shape[1] > 2:
    #    from sklearn.manifold import TSNE
    #    vecs = TSNE(n_components=2).fit_transform(vecs)
#
    #x = list()
    #i = 0
    #for index, row in gold_mapping.iterrows():
    #    x.append([row['gold_src_id'], row['gold_src_id']] + vecs[i].tolist() + vecs[i + 1].tolist())
    #    i = i + 2
    #x = pd.DataFrame(x)
    #x.columns = ['gold_src_id', 'gold_tgt_id', 'src_0', 'src_1', 'tgt_0', 'tgt_1']
#
    #fig = plt.figure()
    #ax = fig.add_subplot(221)
#
    #for index, row in x.iterrows():
    #    origin = np.array(row[['src_0', 'src_1']])
    #    target = np.array(row[['tgt_0', 'tgt_1']]) - origin
    #    ax.arrow(origin[0], origin[1], target[0], target[1], width=0.00005, head_width=0.005, head_length=0.01,
    #             color="green")  # [c[0] for c in np.random.rand(3,1).tolist()])
    #    plt.plot(origin[0], origin[1], 'ro', color="blue")
    #    plt.plot(row[['tgt_0']], row[['tgt_1']], 'ro', color="grey")
#
    #plt.xlim(min(x['src_0'].min(), x['tgt_0'].min()) - 0.1, max(x['src_0'].max(), x['tgt_0'].max()) + 0.1)
    #plt.ylim(min(x['src_1'].min(), x['tgt_1'].min()) - 0.1, max(x['src_1'].max(), x['tgt_1'].max()) + 0.1)
    #fig.savefig(CONFIGURATION.rundir + "embedding_alignments.png", dpi=1000)
    #plt.close()



    # Reduce dimensionality
    gold_mapping = pd.read_csv(CONFIGURATION.gold_mapping.raw_trainsets[0], delimiter="\t", header=None, skiprows=1)
    gold_mapping = gold_mapping.sample(n=min(20, len(gold_mapping)))
    gold_mapping = gold_mapping.applymap(lambda s: s.lower() if type(s) == str else s)
    gold_mapping.columns = ["gold_src_id", "gold_tgt_id", "label"]

    vecs = list()
    ids1 = np.random.choice(np.array(list(graph1.elements.keys())), size=min(100,len(graph1.elements.keys())))
    ids2 = np.random.choice(np.array(list(graph2.elements.keys())), size=min(100,len(graph2.elements.keys())))
    ids = np.concatenate((np.concatenate((np.concatenate((ids1, ids2), axis=0), gold_mapping.gold_src_id.as_matrix()),
                                         axis=0), gold_mapping.gold_tgt_id.as_matrix()), axis=0)
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


    plt.figure(figsize=(40,40))
    for i, label in enumerate(ids):
            x, y = vecs[i,0], vecs[i,1]
            if i < len(ids1):
                plt.scatter(x, y, color='blue', s=500)
            elif i < len(ids1) + len(ids2):
                plt.scatter(x, y, color='red', s=500)
            elif i < len(ids1) + len(ids2) + len(gold_mapping):
                origin = np.array([x,y])
                x2, y2 = vecs[i+len(gold_mapping), 0], vecs[i+len(gold_mapping), 1]
                target = np.array([x2,y2]) - origin
                plt.arrow(origin[0], origin[1], target[0], target[1], linewidth=3, width=0.0001, head_width=0.005, head_length=0.01,
                         color="grey")  # [c[0] for c in np.random.rand(3,1).tolist()])
                plt.scatter(origin[0], origin[1], color='blue', s=500)
                plt.scatter(x2, y2, color='red', s=500)
                #plt.plot(origin[0], origin[1], 'ro', color="blue")
                #plt.plot(x2, y2, 'ro', color="red")

    plt.legend([Line2D([0], [0], color='blue', lw=4), Line2D([0], [0], color='red', lw=4)], ['Graph1', 'Graph2'])
    plt.savefig(CONFIGURATION.rundir+"embeddings_alignments.png")
    plt.close()
