import logging
from externalknowledge.duckduckgo import generate_text
from tokenization.spacytoken import tokenize

logger = logging.getLogger(__name__)


def __repeat_resource_after_every_nth_token(sentences, resource, n):
    modified_list = []
    for sentence in sentences:
        new_sentence = []
        for i, element in enumerate(sentence):
            new_sentence.append(element)
            if i % n == n-1:
                new_sentence.append(resource)
        modified_list.append(new_sentence)
    return modified_list


def generate_literal_walks(kg, window_size_for_repetition=None, text_link_amount=None,
                           sentence_wise=True, with_fragments=True):
    """ generates walks with text from literals """
    for s, p, o in kg.get_literal_triples_with_fragments() if with_fragments else kg.get_literal_triples():
        sentences = list(tokenize(o))
        if window_size_for_repetition:
            sentences = __repeat_resource_after_every_nth_token(sentences, s, window_size_for_repetition)

        if sentence_wise:
            for sent in sentences:
                yield [s, p, *sent]
        else:
            yield [s, p, *[token for sentence in sentences for token in sentence]]

        if text_link_amount is not None:
            for sent in sentences:
                for additional_sent in generate_text(' '.join(sent), amount_links=text_link_amount):
                    yield additional_sent

# if __name__ == '__main__':
#     CONFIGURATION.log(__repeat_resource_after_every_nth_token([['a','b','c','d','e','f','g','h','i','j']], 'x', 3))
