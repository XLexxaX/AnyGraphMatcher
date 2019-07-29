import numpy as np
import logging

logger = logging.getLogger(__name__)

#mainly from https://gist.github.com/bwhite/3726239


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def hits_at_k(rs, k):
    assert k >= 1
    return np.mean([np.asarray(r)[:k].any() for r in rs])


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])

# ==== creation funktions

def create_hits_at_k(k_values=[]):
    return [lambda rs: hits_at_k(rs, k) for k in k_values]

class RankingEvaluation:
    """Computes various ranking evals over multiple datasets (after finishing each dataset the function close_track should be called)"""

    def __init__(self):
        self.rankings = []
        self.track_positions = [0]

    def add_ranking(self, system_ranking, gold):
        self.rankings.append([1 if element == gold else 0 for (element, confidence) in system_ranking ])

    def close_track(self):
        self.track_positions.append(len(self.rankings))

    # def micro_eval(self, eval_func, *args):
    #     return eval_func(self.rankings, *args)
    #
    # def macro_eval(self, eval_func, *args):
    #     if self.track_positions[-1] != len(self.rankings):
    #         logger.warning("Last call to close_track was missing - added it automatically")
    #         self.track_positions.append(len(self.rankings))
    #
    #     track_results = [eval_func(self.rankings[start:stop], *args)
    #                      for start, stop in zip(self.track_positions, self.track_positions[1:])]
    #
    #     return np.mean(track_results)

    def get_eval(self, ranking_metric_function, macro=True):
        if macro:
            if self.track_positions[-1] != len(self.rankings):
                logger.warning("Last call to close_track was missing - added it automatically")
                self.track_positions.append(len(self.rankings))

            track_results = [ranking_metric_function(self.rankings[start:stop])
                             for start, stop in zip(self.track_positions, self.track_positions[1:])]

            return np.mean(track_results)
        else:
            ranking_metric_function(self.rankings)

if __name__ == '__main__':
    test = RankingEvaluation()
    test.add_ranking([('a', 0.49), ('b', 0.49), ('c', 0.49), ('d', 0.49)], 'b')
    test.add_ranking([('a', 0.49), ('b', 0.49), ('c', 0.49), ('d', 0.49)], 'd')
    test.add_ranking([('a', 0.49), ('b', 0.49), ('c', 0.49), ('d', 0.49)], 'd')
    test.close_track()
    test.add_ranking([('a', 0.49), ('b', 0.49), ('c', 0.49), ('d', 0.49)], 'a')
    test.add_ranking([('a', 0.49), ('b', 0.49), ('c', 0.49), ('d', 0.49)], 'e')
    test.close_track()
    CONFIGURATION.log(test.macro_eval(lambda x: hits_at_k(x, 5)))
