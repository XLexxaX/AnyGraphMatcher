import logging
from .dictgenerator import generate_subject_predicate_object_dict

logger = logging.getLogger(__name__)


def generate_all_walks(kg, maximum_length_of_a_walk=50):
    sub_pred_obj_dict = generate_subject_predicate_object_dict(kg)
    subject_count = len(sub_pred_obj_dict.keys())
    for i, resource in enumerate(sub_pred_obj_dict.keys()):
        stack = [(False, None, resource), (True, None, resource)]
        node_list, node_set = [], set()
        path = []  # node, edge, node, edge, node, edge .....
        depth = 0
        while stack:
            enter_or_return, ingoing_edge, node = stack.pop()
            if enter_or_return:  # entering the function
                depth += 1
                node_list.append(node)
                node_set.add(node)
                path.extend((ingoing_edge, node))
                if depth > maximum_length_of_a_walk:
                    yield path[1:]
                else:
                    pred_obj_dict = sub_pred_obj_dict.get(node)
                    # if there are no new neighbors, write out the path
                    if not pred_obj_dict:
                        yield path[1:]
                    else:
                        # add new neighbors
                        for predicate, objects in pred_obj_dict.items():
                            for obj in objects:
                                if obj not in node_set:  # we don't want cycles
                                    stack.append((False, predicate, obj))
                                    stack.append((True, predicate, obj))
            else:  # returning back from the function
                node_set.remove(node_list.pop())
                path.pop()  # node
                path.pop()  # edge
                depth -= 1
        if i % 1000 == 0:
            logger.info("%s / %s", i, subject_count)