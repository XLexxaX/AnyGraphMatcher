import nltk
import logging
from tokenization.camelcase import camel_case_split
logger = logging.getLogger(__name__)


def tokenize(text, lowercase=True):
    if lowercase:
        for sentence in nltk.sent_tokenize(camel_case_split(text).replace('/', ' / ').replace('-', ' - ')):
            yield [w.lower() for w in nltk.word_tokenize(sentence)] # yield a tokenized sentence
    else:
        for sentence in nltk.sent_tokenize(camel_case_split(text).replace('/', ' / ').replace('-', ' - ')):
            yield nltk.word_tokenize(sentence) # yield a tokenized sentence

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
    #logging.info("Start")
    # CONFIGURATION.log(list(tokenize("""
    # A convention, in the sense of a meeting, is a gathering of individuals who meet at an arranged place and time in order to discuss or engage in some common interest.
    # The most common conventions are based upon industry, profession, and fandom.
    # Trade conventions typically focus on a particular industry or industry segment, and feature keynote speakers, vendor displays, and other information and activities of interest to the event organizers and attendees.
    # Professional conventions focus on issues of concern along with advancements related to the profession.
    # """)))
    #
    # for i in tokenize("ThisIsATest and so on and so furth."):
    #     CONFIGURATION.log(i)

    CONFIGURATION.log(list(tokenize("Meta-Gutachater (Gutachter der Gutachten begutachtet)")))

    #CONFIGURATION.log(list())