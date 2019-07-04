import logging

from collections import defaultdict
from cle.loadkg.loadFromDbpedia import load_kg_from_dbpedia
from cle.sentencegenerator.randomwalks import write_random_walks
import os
import re
from itertools import chain

logger = logging.getLogger(__name__)
package_directory = os.path.dirname(os.path.abspath(__file__))


def map_to_language(seventh_eighth_character_of_url):
    """Map the 7. and 8. character in the url (http://de.dbpedia.org/...) to the language.
    In case of english (http://dbpedia.org/resource/..) this is 'db' maps to en."""

    if seventh_eighth_character_of_url == 'db':
        return 'en'
    else:
        return seventh_eighth_character_of_url

def get_count_relationships(language):
    source_dbpedia_path = os.path.join(package_directory, '..', '..', 'data', 'dbpedia', language)
    iterate_object_triples, _ = load_kg_from_dbpedia(source_dbpedia_path, language, infobox_properties=True, max_triples=10000)
    # count relationship triples per resource
    count_relationships_per_resource = defaultdict(int)
    for s, p, o in iterate_object_triples:
        count_relationships_per_resource[s] += 1
    return count_relationships_per_resource


def get_inter_language_links_for_given_languages(set_of_language_pairs):
    for language in set(chain.from_iterable((src, dst) for src, dst in set_of_language_pairs)):
        dbpedia_path = os.path.join(package_directory, '..', '..', 'data', 'dbpedia', language)
        iterate_object_triples, _ = load_kg_from_dbpedia(dbpedia_path, language,interlanguage_links=True, max_triples=10000)
        for s, p, o in iterate_object_triples:
            s_language = map_to_language(s[7:9])
            o_language = map_to_language(o[7:9])
            if (s_language, o_language) in set_of_language_pairs:
                yield s, o, s_language, o_language
            elif (o_language, s_language) in set_of_language_pairs:
                yield o, s, o_language, s_language


def get_inter_language_links_given_min_amount_relationships(set_of_language_pairs, number_of_relationships):
    all_languages = set(chain.from_iterable((src, dst) for src, dst in set_of_language_pairs))
    count_relationship_dict = {lang : get_count_relationships(lang) for lang in all_languages }
    for src, dst, src_lang, dst_lang in get_inter_language_links_for_given_languages(set_of_language_pairs):
        if count_relationship_dict[src_lang][src] > number_of_relationships and count_relationship_dict[dst_lang][dst] > number_of_relationships:
            yield src, dst, src_lang, dst_lang


def save_inter_language_links_given_min_amount_relationships(set_of_language_pairs, number_of_relationships):

    inter_language_links = defaultdict(set)
    for src, dst, src_lang, dst_lang in get_inter_language_links_given_min_amount_relationships(set_of_language_pairs,number_of_relationships):
        inter_language_links[(src_lang, dst_lang)].add((src, dst))

    for language_pair, ill in inter_language_links.items():
        filename = os.path.join(package_directory, '..', '..', 'data', 'multilingual_dbpedia', language_pair[0] + '_' + language_pair[1], 'reference.txt')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as out_file:
            for src, dst in ill:
                out_file.write("{}\t{}".format(src, dst))





def get_property_mapping_from_file(file_path):
    regex_string = r'\s*'.join(
        ['{{', 'PropertyMapping', '\|', 'templateProperty', '=', '(.*?)', '\|', 'ontologyProperty', '=', '(.*?)',
         '(?:\|.*)?', '}}'])
    pat = re.compile(regex_string)
    ont_prop_to_lang_prop = defaultdict(set) # ontology property to property namespace
    with open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            for match in pat.findall(line):
                if match[0] and match[1]: # both non empty strings
                    ont_prop_to_lang_prop[match[1]].add(match[0])
    return ont_prop_to_lang_prop



def get_property_mapping(set_of_language_pairs):

    for src_dst_language in set_of_language_pairs:
        source_prop_mapping = get_property_mapping_from_file(os.path.join(package_directory, '..', '..', 'data', 'dbpedia', src_dst_language[0], 'Mapping_'+ src_dst_language[0] + '.xml'))
        target_prop_mapping = get_property_mapping_from_file(os.path.join(package_directory, '..', '..', 'data', 'dbpedia', src_dst_language[1], 'Mapping_' + src_dst_language[1] + '.xml'))

        print(source_prop_mapping)
        print(target_prop_mapping)

        for common_ont_prop in set(source_prop_mapping.keys()).intersection(target_prop_mapping.keys()):
            logger.info("%s -> %s | %s", common_ont_prop, source_prop_mapping[common_ont_prop], target_prop_mapping[common_ont_prop])






if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
    logging.info("Start")
    language_pairs = set([('en', 'fr')])

    save_inter_language_links_given_min_amount_relationships(language_pairs, 4)

    #get_inter_language_links_given_min_amount_relationships(language_pairs, 4)
    #print(language_pairs)
    #get_property_mapping(language_pairs)

    #get_inter_language_links_given_min_amount_relationships(language_pairs, 4)




