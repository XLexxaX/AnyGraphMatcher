from rdflib import Graph, URIRef, Literal
from xopen import xopen
from urllib.parse import urlparse
from collections import Counter

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
            #print(prefix, url)

def load_kg_from_xml(path):
    g = Graph()
    with xopen(path, 'rb') as f:
        g.parse(f, format="xml")

    #test = __get_namespace(g)
    #test = list(g.namespaces())

    return __yield_object(g), __yield_literal(g)

def load_kg_from_xml_interface(dummy, path):
    return load_kg_from_xml(path)
