import random
import logging
from .dictgenerator import generate_subject_predicate_object_dict

logger = logging.getLogger(__name__)


def __get_next_pred_obj(sub_pred_obj_dict, current_resource):
    pred_obj_map = sub_pred_obj_dict.get(current_resource)
    if pred_obj_map is None:
        return None, None
    chosen_predicate = random.choice(list(pred_obj_map.keys()))
    objects = pred_obj_map[chosen_predicate]
    chosen_object = next(iter(random.sample(objects, 1)))
    return chosen_predicate, chosen_object


def generate_random_walks(kg, number_of_walks_per_resource=10,
                          maximum_length_of_a_walk=50, only_unique_walks=True):
    sub_pred_obj_dict = generate_subject_predicate_object_dict(kg)
    subject_count = len(sub_pred_obj_dict.keys())
    for i, resource in enumerate(sub_pred_obj_dict.keys()):
        walks_per_resource = []
        for k in range(number_of_walks_per_resource):
            current = resource
            one_walk = [current]
            for l in range(maximum_length_of_a_walk):
                (chosen_predicate, chosen_object) = __get_next_pred_obj(sub_pred_obj_dict, current)
                if chosen_predicate is None or chosen_object is None:
                    break
                one_walk.append(chosen_predicate)
                one_walk.append(chosen_object)
                current = chosen_object
            walks_per_resource.append(tuple(one_walk))

        for walk in set(walks_per_resource) if only_unique_walks else walks_per_resource:
            yield walk

        if i % 1000 == 0:
            logger.info("%d / %d", i, subject_count)
