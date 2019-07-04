import logging
from cle.tokenization.spacytoken import tokenize
from cle.externalknowledge.duckduckgo import generate_text
logger = logging.getLogger(__name__)


def generate_literal_walks_with_text(kg):
    for s, p, o in kg.get_literal_triples():

        for sent in tokenize(o):
            yield [s, p, *list(sent)]

            for additional_sent in generate_text(' '.join(sent), amount_links=5):
                yield additional_sent

        # tokenized_object = [for sent in tokenize(o)
        # bal = [s, p, *list(tokenized_object)]
        # yield [s, p, *list(tokenized_object)]
        #
        # for sent in generate_text(tokenized_object):
        #     yield sent


