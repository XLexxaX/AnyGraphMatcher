from cle.wordembedding import EmbeddingsInterface
from cle.configurations.PipelineTools import PipelineDataTuple

import os
import sys
global CONFIGURATION

def interface(main_input, args, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    graph1 = main_input.get(0)
    graph2 = main_input.get(1)
    assert graph1 is not None, "Graph not found in " + os.path.basename(sys.argv[0])
    if graph2 is None:
        return execute(graph1)
    else:
        return PipelineDataTuple(execute(graph1).elems[0], execute(graph2).elems[0])

def execute(graph):
    for descriptor, resource in graph.elements.items():
        tmp = list()
        for embedding in resource.embeddings:
            for num in embedding:
                tmp = tmp + [num]
        resource.embeddings = [tmp]
    return PipelineDataTuple(graph)
