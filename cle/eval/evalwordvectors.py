import os
import math
from operator import itemgetter


def eval_word_vectors(word_vector, tasks=None, word_sim_dir='data/word-sim'):
    '''
    :param word_vector: word embedding as KeyedVector with similarity function
    :param tasks: list of taskname for restriction (e.g. ['EN-MC-30.txt', 'EN-MEN-TR-3k.txt'] )
    :param word_sim_dir: path of directory with similarity data
    :return: list of (task_name, num_word_pair, num_not_found, correlation_spearmans_rho)
    '''
    word_vector.init_sims()
    file_names = os.listdir(word_sim_dir) if tasks is None else tasks
    results = []
    for filename in file_names:
        manual_dict, auto_dict = ({}, {})
        not_found, total_size = (0, 0)
        for line in open(os.path.join(word_sim_dir, filename), 'r'):
            line = line.strip().lower()
            word1, word2, val = line.split()
            if word1 in word_vector and word2 in word_vector:
                manual_dict[(word1, word2)] = float(val)
                auto_dict[(word1, word2)] = word_vector.similarity(word1, word2)
            else:
                not_found += 1
            total_size += 1
        rho = _spearmans_rho(_assign_ranks(manual_dict), _assign_ranks(auto_dict))
        results.append((filename, total_size, not_found, rho))
    return results
        # print "%6s" % str(i+1), "%20s" % filename, "%15s" % str(total_size),
        # print "%15s" % str(not_found),
        # print "%15.4f" % spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict))


def format_eval_results(results_of_eval_word_vectors):
    text ='=================================================================================\n'
    text += "{:>20}{:>15}{:>15}{:>15}\n".format("Dataset", "Num Pairs", "Not found", "Rho")
    text += '=================================================================================\n'
    for (task_name, num_word_pair, num_not_found, correlation_spearmans_rho) in results_of_eval_word_vectors:
        text += "{:>20}{:>15}{:>15}{:15.4f}\n".format(task_name, str(num_word_pair), str(num_not_found), correlation_spearmans_rho)
    return text


def _spearmans_rho(ranked_dict1, ranked_dict2):
    assert len(ranked_dict1) == len(ranked_dict2)
    if len(ranked_dict1) == 0 or len(ranked_dict2) == 0:
        return 0.
    x_avg = 1. * sum([val for val in ranked_dict1.values()]) / len(ranked_dict1)
    y_avg = 1. * sum([val for val in ranked_dict2.values()]) / len(ranked_dict2)
    num, d_x, d_y = (0., 0., 0.)
    for key in ranked_dict1.keys():
        xi = ranked_dict1[key]
        yi = ranked_dict2[key]
        num += (xi - x_avg) * (yi - y_avg)
        d_x += (xi - x_avg) ** 2
        d_y += (yi - y_avg) ** 2
    return num / (math.sqrt(d_x * d_y))


def _assign_ranks(item_dict):
    ranked_dict = {}
    sorted_list = [(key, val) for (key, val) in sorted(item_dict.items(), key=itemgetter(1), reverse=True)]
    for i, (key, val) in enumerate(sorted_list):
        same_val_indices = []
        for j, (key2, val2) in enumerate(sorted_list):
            if val2 == val:
                same_val_indices.append(j + 1)
        if len(same_val_indices) == 1:
            ranked_dict[key] = i + 1
        else:
            ranked_dict[key] = 1. * sum(same_val_indices) / len(same_val_indices)
    return ranked_dict

if __name__ == '__main__':
    from gensim.models import KeyedVectors
    word_vectors = KeyedVectors.load_word2vec_format('global_context.vec', binary=False)
    print(format_eval_results(eval_word_vectors(word_vectors)))