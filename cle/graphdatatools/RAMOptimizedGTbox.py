
from configurations.PipelineTools import PipelineDataTuple
from enum import Enum
from graphdatatools.InvertedIndexToolbox import InvertedIndex, getNGrams
import os
import sys

global CONFIGURATION


class Graph:

    def __init__(self, literal_triples_generator, resource_literals_generator):
        self.objs = list()
        self.lits = list()
        self.type = 'none'

        for s, p, o in spo_generator:
                if not s in self.objs.keys():
                    self.objs[s] = Resource()
                if p.lower() == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
                    self.type = o

                self.objs.append([p,o])
        for s, p, l in spl_generator:
                if not s in self.lits.keys():
                    self.lists[s] = Resource()
                if p.lower() == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
                    self.type = l

                self.lits.append([p,o])




def interface(main_input, args, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    spo_generator = main_input.get(0)
    spl_generator = main_input.get(1)
    assert spo_generator is not None, "S-P-O generator not found in " + os.path.basename(sys.argv[0])
    assert spl_generator is not None, "S-P-L generator not found in " + os.path.basename(sys.argv[0])
    return PipelineDataTuple(Graph(spo_generator, spl_generator))
