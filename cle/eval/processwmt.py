"""Usage: processwmt <type> <file_path>"""
from docopt import docopt
import logging

from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phrases, Phraser

import re

logger = logging.getLogger(__name__)


def multi_replace(text, list_of_characters, replacement=''):
    for elem in list_of_characters:
        if elem in text:
            text = text.replace(elem, replacement)
    return text


replace_characters = ['!', '"', '#', '$', '%', '&', '\\', '\'', '(', ')', '*', '+', '-', '/', ':', ';', '<', '=', '>', '?', '@', '[', ']', '^', '_', '`', '{', '|', '}', '~', '“', '”', '»', '‘', '…']


def preprocess(filepath):
    """Creates corpus remedied for punctuation and numerical variation"""
    with open(filepath, 'rt') as corpus:
        for line in corpus:
            line = multi_replace(line, replace_characters)
            line = re.sub(r"[-+]?\d*\.\d+|\d+", "NUM", line) # replace any number with NUM
            line = re.sub( '\.+', '.', line) # remove multiple dots
            line = ' '.join(line.split())
            if line:
                print(line)


def collocation(in_path):
    """Creates corpus considering collocations, frequent co-occuring bigrams are merged (new york -> new_york)"""
    corpus = LineSentence(in_path)
    bigram = Phraser(Phrases(corpus))
    collocation_corpus = bigram[corpus]
    for sentence in collocation_corpus:
        print(' '.join(sentence))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
    arguments = docopt(__doc__)
    if arguments['<type>'] == 'collocation':
        collocation(arguments['<file_path>'])
    elif arguments['<type>'] == 'preprocess':
        preprocess(arguments['<file_path>'])
    else:
        print(__doc__)
