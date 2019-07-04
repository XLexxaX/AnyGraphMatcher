import logging
import os
import re
import spacy
import mwparserfromhell
from xml.etree.cElementTree import iterparse  # LXML isn't faster, so let's go with the built-in solution

logger = logging.getLogger(__name__)
package_directory = os.path.dirname(os.path.abspath(__file__))



nlp = spacy.load("en_core_web_sm") # en or  en_core_web_sm








def make_url_from_link(link):
    # url has whitespace at the beginning and end because later on at tokenization it is required that each url gets one token
    return ' http://dbpedia.org/resource/' + link.replace(' ', '_')


def get_replacement_list(title, wiki_links):
    replacement_map = dict()
    replacement_map[title.lower()] = make_url_from_link(title)
    for link in wiki_links:
        link_text = link.text or link.title
        replacement_map[link_text.lower()] = make_url_from_link(str(link.title))
    return sorted(list(replacement_map.items()), key=lambda x: len(x[0]), reverse=True)



def process_article(article_text, title):
    parsed_wikicode = mwparserfromhell.parse(article_text)
    text = parsed_wikicode.strip_code()

    text = text.lower()
    for link_text, replacement in get_replacement_list(title, parsed_wikicode.filter_wikilinks()):
        # a lookahead because we dont want to replace "abcdef" when we only have "abc" replace with "123"
        text = re.sub(link_text + '(?![a-zA-Z0-9])', replacement, text)
    print(len(text))
    doc = nlp(text)
    sentences = []
    for sentence in doc.sents:
        tokens = [token.string.strip() for token in sentence if token.string.strip()]
        if len(tokens) > 2:
            sentences.append(' '.join(tokens))

    return sentences


def get_replacement_list_two(title, wiki_links):
    replacement_map = dict()
    replacement_map[title.lower()] = make_url_from_link(title)
    for link in wiki_links:
        link_text = link.text or link.title
        replacement_map[link_text.lower()] = make_url_from_link(str(link.title))

    import hashlib
    import base64
    first = dict()
    second = dict()
    for key, value in replacement_map.items():
        intermediate_text = key.replace(' ', '_')#base64.urlsafe_b64encode(hashlib.md5(key.encode('utf-8')).digest()).decode('utf-8')
        first[key] = intermediate_text
        second[intermediate_text] = value

    first_replace = sorted(list(first.items()), key=lambda x: len(x[0]), reverse=True)
    second_replace = second
    return first_replace, second_replace


def process_article_two(article_text, title):
    parsed_wikicode = mwparserfromhell.parse(article_text)
    text = parsed_wikicode.strip_code()

    text = text.lower()
    first_replace, second_replace = get_replacement_list_two(title, parsed_wikicode.filter_wikilinks())

    for link_text, replacement in first_replace:
        # a lookahead because we dont want to replace "abcdef" when we only have "abc" replace with "123"
        text = re.sub(link_text + '(?![a-zA-Z0-9])', replacement, text)

    doc = nlp(text)
    sentences = []
    for sentence in doc.sents:
        new_sent = []
        for token in sentence:
            stripped_token = token.string.strip()
            if not stripped_token:
                continue
            token = second_replace.get(stripped_token, stripped_token)
            new_sent.append(token)


        sentences.append(' '.join(new_sent))

    return sentences



def get_wiki_text():
    from cle.wikitext.parsewikitextown import get_raw_text_from_markup
    with open(source, 'r', encoding='utf-8') as dump_file, \
         open(target, 'w', encoding='utf-8') as out_file:
        for title, text, pageid in extract_pages(dump_file, filter_namespaces=set(['0'])):
            get_raw_text_from_markup(text)
            #print((title, pageid))
            #for sentence in process_article(text, title):
            #    out_file.write(sentence + '\n')

def run_word2vec():
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence

    model = Word2Vec(LineSentence('tmp.txt'))
    return model.wv



if __name__ == '__main__':
    #logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
    logger.info("Start")
    source = os.path.join(package_directory, '..', '..', 'data', 'test', 'test.xml')  # 'harrypotter_pages_current.xml')
    target = os.path.join(package_directory, '..', '..', 'data', 'test', 'result.txt')
    get_wiki_text()
    #run_word2vec()