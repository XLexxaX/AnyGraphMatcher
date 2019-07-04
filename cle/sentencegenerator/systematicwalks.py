import random
from collections import defaultdict
import logging
from .dictgenerator import generate_subject_predicate_object_dict

logger = logging.getLogger(__name__)


def __distribute_amount_of_walks(elements, amount_of_walks):
    amount_of_walks_for_every_predicate = amount_of_walks // len(elements)
    amount_of_walks_for_random_predicate = amount_of_walks % len(elements)

    predicate_to_amount_of_walks = {predicate: amount_of_walks_for_every_predicate for predicate in elements} \
        if amount_of_walks_for_every_predicate > 0 else defaultdict(int)

    for k in range(amount_of_walks_for_random_predicate):
        predicate_to_amount_of_walks[random.choice(elements)] += 1
    return predicate_to_amount_of_walks


def __remove_objects_resulting_in_cycle(pred_obj_dict, seen_nodes):
    new_dict = defaultdict(set)
    for predicate, objects in pred_obj_dict.items():
        for obj in objects:
            if obj not in seen_nodes:
                new_dict[predicate].add(obj)
    return new_dict


def generate_systematic_runs(kg, number_of_walks_per_resource=20, maximum_length_of_a_walk=50):
    sub_pred_obj_dict = generate_subject_predicate_object_dict(kg)
    subject_count = len(sub_pred_obj_dict.keys())
    for i, resource in enumerate(sub_pred_obj_dict.keys()):
        stack = [(False, None, resource, number_of_walks_per_resource),
                 (True, None, resource, number_of_walks_per_resource)]
        node_list, node_set = [], set()
        path = []  # node, edge, node, edge, node, edge .....
        depth = 0
        while stack:
            enter_or_return, ingoing_edge, node, cur_amount_of_walks = stack.pop()
            if enter_or_return:  # entering the function
                depth += 1
                node_list.append(node)
                node_set.add(node)
                path.extend((ingoing_edge, node))
                if depth > maximum_length_of_a_walk:
                    yield path[1:]
                else:
                    # if there are no new neighbors, write out the path
                    pred_obj_dict = __remove_objects_resulting_in_cycle(sub_pred_obj_dict.get(node, defaultdict(set)), node_set)
                    possible_predicates = list(pred_obj_dict.keys())
                    if len(possible_predicates) == 0:
                        yield path[1:]
                    else:
                        for predicate, pred_amount_of_walks in __distribute_amount_of_walks(possible_predicates, cur_amount_of_walks).items():
                            for obj, object_amount_of_walks in __distribute_amount_of_walks(list(pred_obj_dict[predicate]), pred_amount_of_walks).items():
                                if obj not in node_set:  # we don't want cycles
                                    stack.append((False, predicate, obj, object_amount_of_walks))
                                    stack.append((True, predicate, obj, object_amount_of_walks))
                                else:
                                    logger.error("actual cycle") # todo: remove it before distribuing the amount of walks

            else:  # returning back from the function
                node_set.remove(node_list.pop())
                path.pop()  # node
                path.pop()  # edge
                depth -= 1

        if i % 1000 == 0:
            logger.info("%s / %s", i, subject_count)