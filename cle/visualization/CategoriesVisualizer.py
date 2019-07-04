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

    categories = dict()
    categories_frequency = dict()

    # Reduce dimensionality
    vecs = list()
    ids1 = np.random.choice(np.array(list(graph1.elements.keys())), size=min(10000,len(graph1.elements.keys())))
    ids2 = np.random.choice(np.array(list(graph2.elements.keys())), size=min(10000,len(graph2.elements.keys())))
    ids = np.concatenate((ids1, ids2), axis=0)

    for descriptor in ids:
        try:
            vecs.append(graph1.elements[descriptor].embeddings[0])
            try:
                cat = get_super_category(graph1.elements[descriptor],
                                         'http://rdata2graph.sap.com/hilti_erp/property/t179.id').\
                    literals['http://rdata2graph.sap.com/hilti_erp/property/t179.description']
            except KeyError:
                cat = 'none'
            categories[cat] = [c[0] for c in np.random.rand(3, 1).tolist()]
            if cat in categories_frequency.keys():
                categories_frequency[cat] = categories_frequency[cat] + 1
            else:
                categories_frequency[cat] = 1
        except:
            vecs.append(graph2.elements[descriptor].embeddings[0])
            try:
                cat = get_super_category(graph2.elements[descriptor],
                                         'http://rdata2graph.sap.com/hilti_web/property/categories.id'). \
                    literals['http://rdata2graph.sap.com/hilti_web/property/categories.name']
            except KeyError:
                cat = 'none'
            categories[cat] = [c[0] for c in np.random.rand(3, 1).tolist()]
            if cat in categories_frequency.keys():
                categories_frequency[cat] = categories_frequency[cat] + 1
            else:
                categories_frequency[cat] = 1
    vecs = np.array(vecs)
    #if vecs.shape[1] > 2:
    #    from sklearn.manifold import TSNE
    #    vecs = TSNE(n_components=2).fit_transform(vecs)
    assert vecs.shape[1] == 2, "Visualization mechanism can only work with 2-dimensional embeddings."


    # Randomly sample categories
    #categories_to_show = np.random.choice(np.array(list(categories.keys())), size=25)
    import operator
    categories_to_show = list()
    sorted_x = sorted(categories_frequency.items(), key=operator.itemgetter(1))
    sorted_x = sorted_x[-100:-10]
    sorted_x = np.random.choice(np.array(list(np.array(sorted_x)[:,0])), size=15)
    for x in sorted_x:
        categories_to_show.append(x)



    # Plot embeddings
    plt.figure(figsize=(40,40))
    for i, label in enumerate(ids):
            x, y = vecs[i,0], vecs[i,1]

            if i < len(ids1):
                try:
                    cat = get_super_category(graph1.elements[label],
                                         'http://rdata2graph.sap.com/hilti_erp/property/t179.id').\
                    literals['http://rdata2graph.sap.com/hilti_erp/property/t179.description']
                except KeyError:
                    cat = 'none'
                if cat in categories_to_show:
                    plt.scatter(x, y, color=categories[cat], s=500)
            else:
                try:
                    cat = get_super_category(graph2.elements[label],
                                         'http://rdata2graph.sap.com/hilti_web/property/categories.id'). \
                    literals['http://rdata2graph.sap.com/hilti_web/property/categories.name']
                except KeyError:
                    cat = 'none'
                if cat in categories_to_show:
                    plt.scatter(x, y, color=categories[cat], s=500)

    custom_legend_lines = list()
    custom_legend_labels = list()
    for category, color in categories.items():
        if category in categories_to_show:
            custom_legend_lines.append(Line2D([0], [0], color=color, lw=4))
            custom_legend_labels.append(category)


    plt.legend(custom_legend_lines, custom_legend_labels)

    plt.savefig(CONFIGURATION.rundir+"embeddings_categories.png")
    plt.close()


def get_major_category(node, category_property):
    return get_super_category(node, category_property)
    #try:
    #    node = node.relations[category_property]
    #    return get_super_category(node, category_property)
    #except KeyError:
    #    return node

def get_super_category(node, category_property, i=0):
    try:
        if i>1:
            return node
        i=i+1
        node = node.relations[category_property]
        return get_super_category(node, category_property, i)
    except KeyError:
        return node
