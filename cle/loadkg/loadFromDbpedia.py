import logging
from os import path
from .loadFromNtriples import load_kg_from_ntriples

logger = logging.getLogger(__name__)


def load_kg_from_dbpedia(folder_path, language, labels=False, infobox_properties=False, interlanguage_links=False, max_triples = None):
    mixed_file = []
    only_literals_file = []
    only_objects_file = []


    if labels:
        only_literals_file.append(path.join(folder_path,  'labels_' + language + '.ttl.bz2'))

    if infobox_properties:
        mixed_file.append(path.join(folder_path,  'infobox_properties_' + language + '.ttl.bz2'))

    if interlanguage_links:
        only_objects_file.append(path.join(folder_path,  'interlanguage_links_' + language + '.ttl.bz2'))

    return load_kg_from_ntriples(mixed_file=mixed_file, only_literals_file=only_literals_file,only_objects_file=only_objects_file, max_triples=max_triples)