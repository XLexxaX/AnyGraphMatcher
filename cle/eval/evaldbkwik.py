import re
import random
import math
import logging
import os

from loadkg.kg import KG, KGInMemory
from loadkg.loadFromTargz import load_kg_from_targz_dbkwik_files, load_kg_from_targz_dbkwik_files_in_memory
from loadkg.loadFromXml import load_kg_from_xml
from eval.rankingeval import RankingEvaluation
from eval.confusionmatrix import confusionmatrix
from eval.alignmentformat import parse_mapping_from_file
from eval.rankingeval import create_hits_at_k

logger = logging.getLogger(__name__)
package_directory = os.path.dirname(os.path.abspath(__file__))

dbkwik_class_types = set(['class', 'ontology', 'null'])
dbkwik_prop_types = set(['property', 'ontology', 'null'])
dbkwik_instance_types = set(['resource', 'null'])
type_matcher = re.compile(r"http://dbkwik.webdatacommons.org/.+?/(.+?)/.*")


def __get_type(uri):
    if uri.strip() == 'null':
        return 'null'
    matcher = type_matcher.search(uri)
    if matcher is not None:
        return matcher[1] # first group
    return ''


def __get_class_prop_instance_mappings(mapping):
    class_mapping, prop_mapping, instance_mapping = [], [], []
    for (source, target, relation, confidence) in mapping:
        source_type, target_type = __get_type(source), __get_type(target)
        if source_type in dbkwik_class_types and target_type in dbkwik_class_types:
            class_mapping.append((source, target, relation, confidence))
        elif source_type in dbkwik_prop_types and target_type in dbkwik_prop_types:
            prop_mapping.append((source, target, relation, confidence))
        elif source_type in dbkwik_instance_types and target_type in dbkwik_instance_types:
            instance_mapping.append((source, target, relation, confidence))
    return class_mapping, prop_mapping, instance_mapping


def get_cm_eval_for_class_prop_inst(system_list, gold_list, macro=True):
    class_confusion = confusionmatrix()
    prop_confusion = confusionmatrix()
    inst_confusion = confusionmatrix()
    for system, gold in zip(system_list, gold_list):
        system_class, system_prop, system_instance = __get_class_prop_instance_mappings(system)
        gold_class, gold_prop, gold_instance = __get_class_prop_instance_mappings(gold)

        class_confusion.add_mapping(system_class, gold_class)
        prop_confusion.add_mapping(system_prop, gold_prop)
        inst_confusion.add_mapping(system_instance, gold_instance)
    return class_confusion.get_eval(macro), prop_confusion.get_eval(macro), inst_confusion.get_eval(macro)


def get_cm_eval(system_list, gold_list, macro=True):
    confusion = confusionmatrix()
    for system, gold in zip(system_list, gold_list):
        confusion.add_mapping(system, gold)
    return confusion.get_eval(macro)


def get_ranking_for_class_prop_inst(system_ranking, gold_not_null_list, ranking_metrics=[], macro=True):
    class_ranking_eval = RankingEvaluation()
    prop_ranking_eval = RankingEvaluation()
    inst_ranking_eval = RankingEvaluation()

    for system, gold in zip(system_ranking, gold_not_null_list):
        for i, (src, dst, rel, conf) in enumerate(gold):
            source_type, target_type = __get_type(src), __get_type(dst)
            if source_type in dbkwik_class_types and target_type in dbkwik_class_types:
                class_ranking_eval.add_ranking(system[i], dst)
            elif source_type in dbkwik_prop_types and target_type in dbkwik_prop_types:
                prop_ranking_eval.add_ranking(system[i], dst)
            elif source_type in dbkwik_instance_types and target_type in dbkwik_instance_types:
                inst_ranking_eval.add_ranking(system[i], dst)

        class_ranking_eval.close_track()
        prop_ranking_eval.close_track()
        inst_ranking_eval.close_track()

    return [class_ranking_eval.get_eval(ranking_metric, macro) for ranking_metric in ranking_metrics],\
           [prop_ranking_eval.get_eval(ranking_metric, macro) for ranking_metric in ranking_metrics],\
           [inst_ranking_eval.get_eval(ranking_metric, macro) for ranking_metric in ranking_metrics]


def get_ranking(system_ranking, gold_not_null_list, ranking_metrics=[], macro=True):
    ranking_eval = RankingEvaluation()
    for system, gold in zip(system_ranking, gold_not_null_list):
        for i, (src, dst, rel, conf) in enumerate(gold):
            ranking_eval.add_ranking(system[i], dst)
        ranking_eval.close_track()

    return [ranking_eval.get_eval(ranking_metric, macro) for ranking_metric in ranking_metrics]


def get_wiki_to_file_dict():
    wiki_to_file = dict()
    for f in os.listdir(os.path.join(package_directory, '..', '..', 'data', 'dbkwik', 'KGs_for_gold_standard')):
        splitted_file = f.split('~')
        if len(splitted_file) == 4:
            wiki_to_file[splitted_file[2]] = os.path.join(
                package_directory, '..', '..', 'data', 'dbkwik', 'KGs_for_gold_standard', f)
    return wiki_to_file


def generate_eval_files_dbkwik(selection_sub_tracks=None):
    wiki_to_file = get_wiki_to_file_dict()
    track_folders = os.listdir(os.path.join(package_directory, '..', '..', 'data', 'dbkwik', 'gold'))
    if selection_sub_tracks:
        track_folders = [file_name for file_name in track_folders
                         if file_name.split('~')[0] + '-' + file_name.split('~')[1] in selection_sub_tracks]
    for test_case_folder in track_folders:
        gold_mapping, _, _, _ = parse_mapping_from_file(os.path.join(package_directory, '..', '..', 'data', 'dbkwik', 'gold', test_case_folder))
        src_kg = KGInMemory(lambda: load_kg_from_targz_dbkwik_files_in_memory(wiki_to_file[test_case_folder.split('~')[0]]))
        dst_kg = KGInMemory(lambda: load_kg_from_targz_dbkwik_files_in_memory(wiki_to_file[test_case_folder.split('~')[1]]))
        yield src_kg, dst_kg, gold_mapping


def generate_eval_files_seals(oaei_track, selection_sub_tracks=None):
    track_folders = os.listdir(os.path.join(package_directory, '..', '..', 'data', 'oaei', oaei_track))
    if selection_sub_tracks:
        track_folders = [folder_name for folder_name in track_folders if folder_name in selection_sub_tracks]
    for test_case_folder in track_folders:
        gold_mapping, _, _, _ = parse_mapping_from_file(os.path.join(package_directory, '..', '..', 'data', 'oaei', oaei_track, test_case_folder, 'reference.xml'))
        src_kg = KGInMemory(lambda: load_kg_from_xml(os.path.join(package_directory, '..', '..', 'data', 'oaei', oaei_track, test_case_folder, 'source.xml')))
        dst_kg = KGInMemory(lambda: load_kg_from_xml(os.path.join(package_directory, '..', '..', 'data', 'oaei', oaei_track, test_case_folder, 'target.xml')))
        yield src_kg, dst_kg, gold_mapping


def evaluate(system, track_id, selection_sub_tracks=None,
             initial_mapping_system=None, inital_mapping_gold_standard=None, initial_mapping_file=None,
             percentage_initial_mapping=None, remove_gold=False,
             hits_at_k_values=[1, 5, 10], macro=True):
    max_hits_a_k = max(hits_at_k_values) + 1
    system_list, gold_list = [], []
    system_ranking, gold_not_null_list = [], []

    if track_id == 'dbkwik':
        eval_file_generator = generate_eval_files_dbkwik(selection_sub_tracks)
    else:
        eval_file_generator = generate_eval_files_seals(track_id, selection_sub_tracks)

    for src_kg, dst_kg, gold_mapping in eval_file_generator:
        not_null_gold_mappings = [(src, dst, rel, conf) for (src, dst, rel, conf) in gold_mapping
                                  if src != 'null' and dst != 'null']

        # prepare initial mapping
        initial_mapping = set()
        if initial_mapping_system:
            initial_mapping_system.set_kg(src_kg, dst_kg)
            initial_mapping_system.set_initial_mapping([])
            initial_mapping_system.compute_mapping()
            initial_mapping.update(initial_mapping_system.get_mapping())
        if inital_mapping_gold_standard:
            initial_mapping.update(not_null_gold_mappings)
        if initial_mapping_file:
            parsed_initial_mapping, _, _, _ = parse_mapping_from_file(initial_mapping_file)
            initial_mapping.update(parsed_initial_mapping)

        if remove_gold:
            gold_set = set([(src, dst) for src, dst, rel, conf in not_null_gold_mappings])
            for src, dst, rel, conf in initial_mapping:
                if (src, dst) in gold_set:
                    initial_mapping.remove((src, dst, rel, conf))

        initial_mapping = list(initial_mapping)
        if percentage_initial_mapping:
            initial_mapping = random.sample(initial_mapping,
                                            k=math.ceil(len(initial_mapping) * percentage_initial_mapping))

        #run system
        system.set_kg(src_kg, dst_kg)
        system.set_initial_mapping(initial_mapping)
        system.compute_mapping()

        #eval
        gold_list.append(gold_mapping)
        gold_not_null_list.append(not_null_gold_mappings)
        system_list.append(system.get_mapping())
        system_ranking.append(system.get_mapping_with_ranking([src for src, dst, rel, conf in not_null_gold_mappings],
                                                         topn=max_hits_a_k))

    #final results
    if track_id == 'dbkwik':
        whole_cm_eval = get_cm_eval_for_class_prop_inst(system_list, gold_list, macro)
        whole_ranking_eval = get_ranking_for_class_prop_inst(system_ranking, gold_not_null_list, create_hits_at_k(hits_at_k_values), macro)
        logger.info("{:^41} | {:^41} | {:^41}".format("Classes", "Properties", "Instances"))
        logger.info(' | '.join(["{:^6} {:^6} {:^6} {:^6} {:^6} {:^6}".format(
            "Prec", "Rec", "F-1", "H@1", "H@5", "H@10") for i in range(3)]))
        logger.info(' | '.join(["{:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}".format(
            *[i * 100 for i in list(cm_eval) + list(ranking_eval)]) for cm_eval, ranking_eval in zip(whole_cm_eval, whole_ranking_eval)]))
    else:
        whole_cm_eval = get_cm_eval(system_list, gold_list, macro)
        whole_ranking_eval = get_ranking(system_ranking, gold_not_null_list, create_hits_at_k(hits_at_k_values), macro)

        logger.info("{:^6} {:^6} {:^6} {:^6} {:^6} {:^6}".format("Prec", "Rec", "F-1", "H@1", "H@5", "H@10"))
        logger.info("{:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}".format(*[i * 100 for i in list(whole_cm_eval) + list(whole_ranking_eval)]))

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
    eval_class = ((0.25, 0.5454545454545454, 0.34285714285714286), (1.0, 0.07142857142857142, 0.13333333333333333), (0.8148148148148148, 0.4782608695652174, 0.6027397260273973))
    hits = ([0.63636363636363635, 0.63636363636363635, 0.63636363636363635], [0.071428571428571425, 0.071428571428571425, 0.071428571428571425], [0.47826086956521741, 0.47826086956521741, 0.47826086956521741])
    #CONFIGURATION.log([i * 100 for i in eval_class])
    #CONFIGURATION.log("{:3.2f}->{:3.2f}".format(*[0.25, 0.5454545454545454]))
    logger.info("{:^41} | {:^41} | {:^41}".format("Classes", "Properties", "Instances"))
    logger.info(' | '.join(["{:^6} {:^6} {:^6} {:^6} {:^6} {:^6}".format("Prec", "Rec", "F-1", "H@1", "H@5", "H@10") for i in range(3)]))
    logger.info(' | '.join(["{:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}".format(*[i * 100 for i in list(cm_eval) + list(rank_eval)]) for cm_eval, rank_eval in zip(eval_class, hits)]))

    #logger.info("{:^6} {:^6} {:^6} | {:^6} {:^6} {:^6} | {:^6} {:^6} {:^6}".format("Prec", "Rec", "F-1", "Prec", "Rec", "F-1", "Prec", "Rec", "F-1"))
    #logger.info("{:6.2f} {:6.2f} {:6.2f} | {:6.2f} {:6.2f} {:6.2f} | {:6.2f} {:6.2f} {:6.2f}".format(*[item * 100 for sublist in eval_class for item in sublist]))
    #logger.info(hits)
    #logger.info('{:06.2f}'.format(3.141592653589793))