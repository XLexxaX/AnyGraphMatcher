import tarfile
from nparser import parse, Resource, Literal
from itertools import chain
import logging


logger = logging.getLogger(__name__)


def yield_object_and_literals(file_path, name):
    with tarfile.open(file_path, encoding='utf8') as tar:
        try:
            for i, (s, p, o) in enumerate(parse(tar.extractfile(name))):
                yield s.value, p.value, o.value
                if i % 1000000 == 0:
                    logger.info("File %s in file %s line %d", name, file_path, i)
        except KeyError:
            logger.error("could not find file {} in {}".format(name, file_path))


def yield_literals_given_mixed(file_path, name):
    with tarfile.open(file_path, encoding='utf8') as tar:
        try:
            for i, (s, p, o) in enumerate(parse(tar.extractfile(name))):
                if type(o) is Literal:
                    yield s.value, p.value, o.value
                if i % 1000000 == 0:
                    logger.info("File %s in file %s line %d", name, file_path, i)
        except KeyError:
            logger.error("could not find file {} in {}".format(name, file_path))


def yield_objects_given_mixed(file_path, name):
    with tarfile.open(file_path, encoding='utf8') as tar:
        try:
            for i, (s, p, o) in enumerate(parse(tar.extractfile(name))):
                if type(o) is Resource:
                    yield s.value, p.value, o.value
                if i % 1000000 == 0:
                    logger.info("File %s in file %s line %d", name, file_path, i)
        except KeyError:
            logger.error("could not find file {} in {}".format(name, file_path))


def restrict_triples(triple_generator, max_triples):
    for i, triple in enumerate(triple_generator):
        if i > max_triples:
            break
        yield triple


def load_kg_from_targz(file_path, language='en', extraction_date='20170801',
                       labels=False, infobox_properties=False, infobox_properties_redirected=False, interlanguage_links=False,
                       article_categories=False, category_labels=False, disambiguations_redirected=False, external_links=False,
                       images=False, infobox_properties_definitions=False, long_abstract=False, short_abstract=False, skos_categories=False,
                       template_type=False, template_type_definition=False,
                       max_triples=None):
    base_file_name = "{}wiki-{}-".format(language, extraction_date)
    object_generators = []
    literal_generators = []

    if labels:
        literal_generators.append(yield_object_and_literals(file_path, base_file_name + 'labels.ttl'))
    if infobox_properties:
        literal_generators.append(yield_literals_given_mixed(file_path, base_file_name + 'infobox-properties.ttl'))
        object_generators.append(yield_objects_given_mixed(file_path, base_file_name + 'infobox-properties.ttl'))
    if infobox_properties_redirected:
        literal_generators.append(yield_literals_given_mixed(file_path, base_file_name + 'infobox-properties-redirected.ttl'))
        object_generators.append(yield_objects_given_mixed(file_path, base_file_name + 'infobox-properties-redirected.ttl'))
    if interlanguage_links:
        object_generators.append(yield_object_and_literals(file_path, base_file_name + 'interlanguage-links.ttl'))
    if article_categories:
        object_generators.append(yield_object_and_literals(file_path, base_file_name + 'article-categories.ttl'))
    if category_labels:
        literal_generators.append(yield_object_and_literals(file_path, base_file_name + 'category-labels.ttl'))
    if disambiguations_redirected:
        object_generators.append(yield_object_and_literals(file_path, base_file_name + 'disambiguations-redirected.ttl'))
    if external_links:
        object_generators.append(yield_object_and_literals(file_path, base_file_name + 'external-links.ttl'))
    if images:
        object_generators.append(yield_object_and_literals(file_path, base_file_name + 'images.ttl'))
    if infobox_properties_definitions:
        literal_generators.append(yield_literals_given_mixed(file_path, base_file_name + 'infobox-property-definitions.ttl'))
        object_generators.append(yield_objects_given_mixed(file_path, base_file_name + 'infobox-property-definitions.ttl'))
    if long_abstract:
        literal_generators.append(yield_object_and_literals(file_path, base_file_name + 'long-abstracts.ttl'))
    if short_abstract:
        literal_generators.append(yield_object_and_literals(file_path, base_file_name + 'short-abstracts.ttl'))
    if skos_categories:
        literal_generators.append(yield_literals_given_mixed(file_path, base_file_name + 'skos-categories.ttl'))
        object_generators.append(yield_objects_given_mixed(file_path, base_file_name + 'skos-categories.ttl'))
    if template_type:
        object_generators.append(yield_object_and_literals(file_path, base_file_name + 'template-type.ttl'))
    if template_type_definition:
        literal_generators.append(yield_literals_given_mixed(file_path, base_file_name + 'template-type-definitions.ttl'))
        object_generators.append(yield_objects_given_mixed(file_path, base_file_name + 'template-type-definitions.ttl'))

    if max_triples is not None:
        literal_generators = [restrict_triples(gen, max_triples) for gen in literal_generators]
        object_generators = [restrict_triples(gen, max_triples) for gen in object_generators]

    return chain.from_iterable(object_generators), chain.from_iterable(literal_generators)


def load_kg_from_targz_dbkwik_files(file_path, max_triples=None):
    return load_kg_from_targz(file_path,
                              article_categories=True, category_labels=True, disambiguations_redirected=True,
                              external_links=True, images=True, infobox_properties_redirected=True, infobox_properties_definitions=True,
                              labels=True, long_abstract=True, short_abstract=True, skos_categories=True, template_type=True, template_type_definition=True,
                              max_triples=max_triples)


def load_kg_from_targz_dbkwik_files_in_memory(file_path, language='en', extraction_date='20170801', max_triples=None):
    literal_stat = []
    obj_stat = []

    base_file_name = "{}wiki-{}-".format(language, extraction_date)
    with tarfile.open(file_path, encoding='utf8') as tar:
        #['article-categories.ttl', 'category-labels.ttl', 'disambiguations-redirected.ttl', 'external-links.ttl', 'images.ttl', 'infobox-properties-redirected.ttl',
        #             'infobox-property-definitions.ttl', 'labels.ttl', 'long-abstracts.ttl', 'short-abstracts.ttl', 'skos-categories.ttl', 'template-type.ttl', 'template-type-definitions.ttl']:
        for name in ['infobox-properties-redirected.ttl', 'template-type.ttl', 'template-type-definitions.ttl', 'infobox-property-definitions.ttl', 'labels.ttl']:
            try:
                for i, (s, p, o) in enumerate(parse(tar.extractfile(base_file_name + name))):
                    if type(o) is Literal:
                        literal_stat.append((s.value, p.value, o.value))
                    else:
                        obj_stat.append((s.value, p.value, o.value))
                    if i % 1000000 == 0:
                        logger.info("File %s in file %s line %d", name, file_path, i)
            except KeyError:
                logger.error("could not find file {} in {}".format(name, file_path))
    return (n for n in obj_stat), (n for n in literal_stat)