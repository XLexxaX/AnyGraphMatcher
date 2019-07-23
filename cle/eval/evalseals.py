import os

from eval.alignmentformat import parse_mapping_from_file
import logging
import os
from loadkg.kg import KG
from loadkg.loadFromXml import load_kg_from_xml


def eval_seals(folder, system_function):
    system_list, gold_list = [], []
    for test_case_folder in os.listdir(folder):
        gold_mapping, _, _, _ = parse_mapping_from_file(os.path.join(folder, test_case_folder, 'reference.xml'))
        gold_list.append(gold_mapping)
        src_kg = KG(lambda: load_kg_from_xml(os.path.join(folder, test_case_folder, 'source.xml')))
        dst_kg = KG(lambda: load_kg_from_xml(os.path.join(folder, test_case_folder, 'target.xml')))
        system_list.append(system_function(src_kg, dst_kg))


