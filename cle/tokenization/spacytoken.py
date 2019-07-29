import spacy
import logging
from tokenization.camelcase import camel_case_split
logger = logging.getLogger(__name__)
import re

# def tokenize(text):
#     #nlp = spacy.load("en")
#     #doc = nlp(text, disable=['parser', 'tagger', 'ner'])
#     nlp = spacy.blank('en')
#     nlp.add_pipe(nlp.create_pipe('sentencizer'))
#     doc = nlp(text)
#     test = list(doc)
#     for token in doc:
#         CONFIGURATION.log(token)
#     for l in doc.sents:
#         CONFIGURATION.log(l)

nlp = spacy.load("en")



def tokenize(text, lowercase=True):
    #https://spacy.io/usage/linguistic-features#section-sbd
    #rule based:
    #nlp = spacy.blank('en')
    #nlp.add_pipe(nlp.create_pipe('sentencizer'))

    #Default: Using the dependency parse
    doc = nlp(camel_case_split(text))
    if lowercase:
        for sentence in doc.sents:
            yield [token.string.strip().lower() for token in sentence]
    else:
        for sentence in doc.sents:
            yield [token.string.strip() for token in sentence]

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
    #logging.info("Start")

    test = list(tokenize("Right_Coronary_Artery"))
    CONFIGURATION.log(test)
    # for i in tokenize("""A convention, in the sense of a meeting, is a gathering of individuals who meet at an arranged place and time in order to discuss or engage in some common interest and there is a camelCaseWordInside here.
    # The most common conventions are based upon industry, profession, and fandom.
    # Trade conventions typically focus on a particular industry or industry segment, and feature keynote speakers, vendor displays, and other information and activities of interest to the event organizers and attendees.
    # Professional conventions focus on issues of concern along with advancements related to the profession.
    # """):
    #     CONFIGURATION.log(i)

    #tokenize("ThisIsATest and so on and so furth.")
