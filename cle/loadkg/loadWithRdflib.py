from rdflib import Graph, URIRef, Literal
from urllib.parse import urlparse
from collections import Counter
from configurations.PipelineTools import PipelineDataTuple
import os
import sys

def __yield_object(g):
    for s, p, o in g:
        if type(o) is URIRef:
            yield (str(s), str(p), str(o))


def __yield_literal(g):
    for s, p, o in g:
        if type(o) is Literal:
            yield (str(s), str(p), str(o))


def __get_namespace(graph):
    for prefix, url in graph.namespaces():
        if not prefix.strip():
            return str(url)
    common_namespace = Counter()
    for s, p, o in graph:
        o = urlparse(str(s))
        #common_namespace.update(o.)
            #CONFIGURATION.log(prefix, url)


def load_kg_with_rdflib(path, format=None):
    g = Graph()
    with open(path, 'rb') as f:
        g.parse(f, format=format)

    #test = __get_namespace(g)
    #test = list(g.namespaces())

    return PipelineDataTuple(__yield_object(g), __yield_literal(g))


def load_kg_with_rdflib_nt(path):
    return load_kg_with_rdflib(path, "nt")


def load_kg_with_rdflib_ttl_interface(main_input_mock, args, configuration):
    nt_inputfile = args.get(0)
    assert nt_inputfile is not None, "Path to NT-sourcefile not found in " + os.path.basename(sys.argv[0])
    return load_kg_with_rdflib(nt_inputfile, "ttl")
