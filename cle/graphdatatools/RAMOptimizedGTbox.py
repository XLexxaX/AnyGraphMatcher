
from configurations.PipelineTools import PipelineDataTuple
from enum import Enum
from graphdatatools.InvertedIndexToolbox import InvertedIndex, getNGrams
import os
import sys

global CONFIGURATION

class Resource:
    def __init__(self):
        self.objs = list()
        self.lits = list()
        self.type = 'none'

class Graph:

    def __init__(self, spo_generator, spl_generator):
        self.elements = dict()

        for s, p, o in spo_generator:
                if not s in self.elements.keys():
                    self.elements[s] = Resource()
                if p.lower() == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
                    self.elements[s].type = o

                self.elements[s].objs.append([p,o])
        for s, p, l in spl_generator:
                if not s in self.elements.keys():
                    self.elements[s] = Resource()
                if p.lower() == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
                    self.elements[s].type = l

                self.elements[s].lits.append([p,l])




def interface(main_input, args, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    spo_generator = main_input.get(0)
    spl_generator = main_input.get(1)
    assert spo_generator is not None, "S-P-O generator not found in " + os.path.basename(sys.argv[0])
    assert spl_generator is not None, "S-P-L generator not found in " + os.path.basename(sys.argv[0])
    return PipelineDataTuple(Graph(spo_generator, spl_generator))
