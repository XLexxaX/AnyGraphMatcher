"""
This module computes the Barista embedding from paper "Simple task-specific bilingual word embeddings∗"
http://ruder.io/cross-lingual-embeddings/index.html#randomtranslationreplacement
http://www.aclweb.org/anthology/N15-1157
"""

from random import random, choice
from collections import defaultdict
import itertools
import logging
logger = logging.getLogger(__name__)


def __iter_random(sentences_one, sentences_two):
    while True:
        if random() > 0.5:
            yield next(sentences_one)
        else:
            yield next(sentences_two)


def __get_lexicon_dict(lexicon, both_directions):
    lexicon_dict = defaultdict(set)
    for src_word, dst_word in lexicon:
        lexicon_dict[src_word].add(dst_word)
        if both_directions:
            lexicon_dict[dst_word].add(src_word)
    return {key: list(value) for key, value in lexicon_dict.items()}# lexicon_dict


def __generate_random_translation(sentences_one, sentences_two, lexicon, replacement_chance, both_directions, balance_and_randomize):
    # TODO: maybe more specific to only replace words from same dictionary.
    lexicon_dict = __get_lexicon_dict(lexicon, both_directions)

    if balance_and_randomize:
        combined_iterator = __iter_random(sentences_one, sentences_two)
    else:
        combined_iterator = itertools.chain(sentences_one, sentences_two)
    opposite_replacement_chance = 1 - replacement_chance

    for sentence in combined_iterator:
        new_sentence = []
        for word in sentence:
            possible_replacement = lexicon_dict.get(word)  # TODO: maybe lowercase and strip?
            if possible_replacement is not None and random() > opposite_replacement_chance:
                new_sentence.append(choice(possible_replacement))
            else:
                new_sentence.append(word)
        yield new_sentence


def get_random_translation_embedding(sentences_one, sentences_two, lexicon, embedding_generation_function,
                                     replacement_chance=0.5, both_directions=False, balance_and_randomize=True):
    """
    Computes the Barista embedding.
    :param sentences_one: iterator of iterator like [['word1', 'word2'],['new', 'sentence']]
    :param sentences_two: iterator of iterator like [['word1', 'word2'],['new', 'sentence']]
    :param lexicon: iterable of (source_word, target_word)
    :param embedding_generation_function: the function which takes sentences and generates an embedding (keyed vector)
    :param replacement_chance: chance on replacement of a word
    :param both_directions: use the dictionary in both directions
    :param balance_and_randomize: if true (default) it will balance and romize the dataset by choosing randomly from one or the other source
    :return: generator of sentences - list of tokens(to be used by an embedding)
    """

    return embedding_generation_function(
        __generate_random_translation(sentences_one, sentences_two, lexicon, replacement_chance, both_directions, balance_and_randomize)
    )



