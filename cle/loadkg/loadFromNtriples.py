from nparser import parse, Resource, Literal
from xopen import xopen
from itertools import chain
import logging

logger = logging.getLogger(__name__)


def yield_object_and_literals(full_path):
    with xopen(full_path, 'rb') as f:
        for i, (s, p, o) in enumerate(parse(f)):
            yield s.value, p.value, o.value
            if i % 1000000 == 0:
                logger.info("File %s line %d", full_path, i)


def yield_literals_given_mixed(full_path):
    with xopen(full_path, 'rb') as f:
        for i, (s, p, o) in enumerate(parse(f)):
            if type(o) is Literal:
                yield s.value, p.value, o.value
            if i % 1000000 == 0:
                logger.info("File %s line %d", full_path, i)

def yield_objects_given_mixed(full_path):
    with xopen(full_path, 'rb') as f:
        for i, (s, p, o) in enumerate(parse(f)):
            if type(o) is Resource:
                yield s.value, p.value, o.value
            if i % 1000000 == 0:
                logger.info("File %s line %d", full_path, i)


def restrict_triples(triple_generator, max_triples):
    for i, triple in enumerate(triple_generator):
        if i > max_triples:
            break
        yield triple


def load_kg_from_ntriples(mixed_file=None, only_literals_file=None, only_objects_file=False, max_triples = None):
    object_generators = []
    literal_generators = []

    if mixed_file:
        for file_path in mixed_file:
            object_generators.append(yield_objects_given_mixed(file_path))
            literal_generators.append(yield_literals_given_mixed(file_path))

    if only_literals_file:
        for file_path in only_literals_file:
            literal_generators.append(yield_object_and_literals(file_path))

    if only_objects_file:
        for file_path in only_objects_file:
            object_generators.append(yield_object_and_literals(file_path))

    if max_triples is not None:
        literal_generators = [restrict_triples(gen, max_triples) for gen in literal_generators]
        object_generators = [restrict_triples(gen, max_triples) for gen in object_generators]

    return chain.from_iterable(object_generators), chain.from_iterable(literal_generators)
