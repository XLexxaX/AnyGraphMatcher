from collections import defaultdict
import logging
import sys

logger = logging.getLogger(__name__)

class confusionmatrix():
    def __init__(self):
        self.tp = []
        self.fp = []
        self.fn = []

    def add_mapping(self, system, gold):
        system_map_source_target = defaultdict(set)
        system_map_target_source = defaultdict(set)
        for (source, target, relation, confidence) in system:
            if relation == '=':
                system_map_source_target[source].add(target)
                system_map_target_source[target].add(source)

        true_positive = 0
        false_positive = 0
        false_negative = 0
        for (source, target, relation, confidence) in gold:
            if target == 'null':
                false_positive += len(system_map_source_target.get(source, set()))
                if len(system_map_source_target.get(source, set())) > 0:
                    logger.debug("too much {} = {}".format(source, system_map_source_target.get(source, set())))
            elif source == 'null':
                false_positive += len(system_map_target_source.get(target, set()))
                if len(system_map_target_source.get(target, set())) > 0:
                    logger.debug("too much {} = {}".format(system_map_target_source.get(target, set()), target))
            else:
                system_targets = system_map_source_target.get(source, set())
                system_sources = system_map_target_source.get(target, set())

                if target in system_targets:
                    true_positive += 1
                    logger.debug("true_positiv {} = {}".format(source, target))
                    false_positive += len(system_sources) - 1
                    false_positive += len(system_targets) - 1
                    if len(system_sources) > 1:
                        logger.debug("too much: ".format(system_sources))
                    if len(system_targets) > 1:
                        logger.debug("too much: ".format(system_targets))
                else:
                    false_negative += 1
                    logger.debug("not found {} = {}".format(source, target))
                    false_positive += len(system_sources)
                    if len(system_sources) > 0:
                        logger.debug("too much: {} = {}".format(system_sources, target))
                    false_positive += len(system_targets)
                    if len(system_targets) > 0:
                        logger.debug("too much: {} = {}".format(source, system_targets))

        self.tp.append(true_positive)
        self.fp.append(false_positive)
        self.fn.append(false_negative)

    def __micro_eval(self):
        true_positive = sum(self.tp)
        false_positive = sum(self.fp)
        false_negative = sum(self.fn)

        precision = self.divide_with_two_denominators(true_positive, true_positive, false_positive)
        recall = self.divide_with_two_denominators(true_positive, true_positive, false_negative)

        return precision, recall, self.divide_with_two_denominators((2.0 * recall * precision), recall, precision)

    def __macro_eval(self):
        precision = 0.0
        recall = 0.0
        for true_positive, false_positive, false_negative in zip(self.tp, self.fp, self.fn):
            precision += self.divide_with_two_denominators(true_positive, true_positive, false_positive)
            recall += self.divide_with_two_denominators(true_positive, true_positive, false_negative)
        precision /= max(1, len(self.tp), len(self.fp), len(self.fn))
        recall /= max(1, len(self.tp), len(self.fp), len(self.fn))

        return precision, recall, self.divide_with_two_denominators((2.0 * recall * precision), recall, precision)


    def get_eval(self, macro=True):
        if macro:
            return self.__macro_eval()
        else:
            return self.__micro_eval()

    def divide_with_two_denominators(self, numerator, denominator_one, denominator_two):
        if denominator_one + denominator_two > 0:
            return numerator / (denominator_one + denominator_two)
        else:
            return 0.0