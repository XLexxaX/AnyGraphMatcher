from cle.sentencegenerator.readfileutil import read_from_file
from cle.configurations.PipelineTools import PipelineDataTuple
import os
import sys
global CONFIGURATION

def interface(main_input, args, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    graph1 = main_input.get(0)
    corpus_file = args.get(0)
    properties = args.get(1)
    assert graph1 is not None, "Graph not found in " + os.path.basename(sys.argv[0])
    assert corpus_file is not None, "Path to corpus file not found in " + os.path.basename(sys.argv[0])
    return execute(graph1, corpus_file, properties)

def execute(graph1, corpus_file, properties):
    graph1.corpus = read_from_file(corpus_file, properties)
    return PipelineDataTuple(graph1)
