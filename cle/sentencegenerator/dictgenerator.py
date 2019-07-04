from collections import defaultdict


def generate_subject_predicate_object_dict(kg):
    sub_pred_obj_dict = defaultdict(lambda: defaultdict(set))
    for s, p, o in kg.get_object_triples():
        sub_pred_obj_dict[s][p].add(o)
    return sub_pred_obj_dict
