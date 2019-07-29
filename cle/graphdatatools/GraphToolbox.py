
from configurations.PipelineTools import PipelineDataTuple
from enum import Enum
from graphdatatools.InvertedIndexToolbox import InvertedIndex, getNGrams
import os
import sys

global CONFIGURATION

class Type(Enum):

    class CLASS:
        def check(self, resource):
            if 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type' in resource.relations.keys():
                if "http://www.w3.org/2000/01/rdf-schema" == resource.descriptor:
                    return True
            return False

        def to_string(self):
            return 'CLASS'

    class TABLE:
        def check(self, resource):
            if 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type' in resource.relations.keys():
                y = resource.relations['http://www.w3.org/1999/02/22-rdf-syntax-ns#type']
                if "http://www.w3.org/2000/01/rdf-schema#class" == resource.relations['http://www.w3.org/1999/02/22-rdf-syntax-ns#type'].descriptor:
                    return True
            return False

        def to_string(self):
            return 'TABLE'

    class PROPERTY:
        def check(self, resource):
            if 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type' in resource.relations.keys():
                if "http://www.w3.org/2000/01/rdf-schema#property" == resource.relations['http://www.w3.org/1999/02/22-rdf-syntax-ns#type'].descriptor:
                    return True
            return False

        def to_string(self):
            return 'PROPERTY'

    class INSTANCE:
        def check(self, resource):
            if not Type.TABLE.value().check(resource) and not Type.CLASS.value().check(resource) \
                    and not Type.PROPERTY.value().check(resource):
                return True
            else:
                return False

        def to_string(self):
            return 'INSTANCE'

    @classmethod
    def determine_type(cls, resource):
        if cls.CLASS.value().check(resource):
            return Type.CLASS.value()
        elif cls.PROPERTY.value().check(resource):
            return Type.PROPERTY.value()
        elif cls.TABLE.value().check(resource):
            return Type.TABLE.value()
        else:
            return Type.INSTANCE.value()




class Resources(dict):

    def create(self, descriptor, etype):
        elem = Resource(descriptor)
        self.__dict__[descriptor] = elem
        return elem

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __unicode__(self):
        return repr(self.__dict__).decode("UTF-8")

    def create_or_get(self, descriptor, type):
        if self.has_key(descriptor):
            return self.__dict__[descriptor]
        else:
            return self.create(descriptor, type)

    def add_resources(self, generator, literal_properties):
        for s, p, o in generator:
            if 'http://dbkwik.webdatacommons.org/oldschoolrunescape/class/Recipe' in s:
                CONFIGURATION.log('here')
            subj = self.create_or_get(s, Resource)
            pred = self.create_or_get(p, Resource)
            obj = self.create_or_get(o, Resource)
            subj.relations[p] = obj
            literal_properties.add(p)

    def add_literals(self, generator, iindex, literal_properties):
        for s, p, l in generator:
            if 'http://dbkwik.webdatacommons.org/oldschoolrunescape/class/Recipe' in s:
                CONFIGURATION.log('here')
            subj = self.create_or_get(s, Resource)
            pred = self.create_or_get(p, Resource)
            try:
                val = subj.literals[p]
            except KeyError:
                val = ""
            subj.literals[p] = val+l
            pred.texts.append(l)
            iindex.addToIndex(val+l, s)
            literal_properties.add(p)


class Resource:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.embeddings = list()
        self.literals = dict()
        self.relations = dict()
        self.texts = list()
        self.type = None
        self.synactic_correspondences = list()


class Graph:

    def __init__(self, literal_triples_generator, resource_literals_generator, file):
        self.file = file
        self.corpus = None
        self.iindex = InvertedIndex()
        self.elements = Resources()
        # Needed for creating an order for levenshtein distance calculation
        self.relation_properties = set()
        self.literal_properties = set()
        self.elements.add_resources(literal_triples_generator, self.relation_properties)
        self.elements.add_literals(resource_literals_generator, self.iindex, self.literal_properties)
        for key in self.elements.keys():
            self.elements[key].type = Type.determine_type(self.elements[key])


    def to_string(self):
        return str(self.file)

def interface(main_input, args, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    nt_filepath = args.get(0)
    spo_generator = main_input.get(0)
    spl_generator = main_input.get(1)
    assert spo_generator is not None, "S-P-O generator not found in " + os.path.basename(sys.argv[0])
    assert spl_generator is not None, "S-P-L generator not found in " + os.path.basename(sys.argv[0])
    assert nt_filepath is not None, "Path to NT-sourcefile not found in " + os.path.basename(sys.argv[0])
    return PipelineDataTuple(Graph(spo_generator, spl_generator, nt_filepath))

