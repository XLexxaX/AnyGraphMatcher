import logging
import os
import pprint
import re

from xml.etree.cElementTree import \
    iterparse  # LXML isn't faster, so let's go with the built-in solution

logger = logging.getLogger(__name__)
package_directory = os.path.dirname(os.path.abspath(__file__))


source = os.path.join(package_directory, '..', '..', 'data', 'test', 'test.xml')#'harrypotter_pages_current.xml')


def get_namespace(tag):
    """Get the namespace of tag.
    Parameters
    ----------
    tag : str
        Namespace or tag.
    Returns
    -------
    str
        Matched namespace or tag.
    """
    m = re.match("^{(.*?)}", tag)
    namespace = m.group(1) if m else ""
    if not namespace.startswith("http://www.mediawiki.org/xml/export-"):
        raise ValueError("%s not recognized as MediaWiki dump namespace" % namespace)
    return namespace


def extract_pages(f, filter_namespaces):
    """Extract pages from a MediaWiki database dump.
    Parameters
    ----------
    f : file
        File-like object.
    filter_namespaces : set of str
         Namespaces that will be extracted.
    Yields
    ------
    tuple of (str or None, str, str)
        Title, text and page id.
    """
    elems = (elem for _, elem in iterparse(f, events=("end",)))

    # We can't rely on the namespace for database dumps, since it's changed
    # it every time a small modification to the format is made. So, determine
    # those from the first element we find, which will be part of the metadata,
    # and construct element paths.
    elem = next(elems)
    namespace = get_namespace(elem.tag)
    ns_mapping = {"ns": namespace}
    page_tag = "{%(ns)s}page" % ns_mapping
    text_path = "./{%(ns)s}revision/{%(ns)s}text" % ns_mapping
    title_path = "./{%(ns)s}title" % ns_mapping
    ns_path = "./{%(ns)s}ns" % ns_mapping
    pageid_path = "./{%(ns)s}id" % ns_mapping

    for elem in elems:
        if elem.tag == page_tag:
            ns = elem.find(ns_path).text
            if ns in filter_namespaces:
                title = elem.find(title_path).text
                pageid = elem.find(pageid_path).text
                text = elem.find(text_path).text
                yield title, text, pageid

            # Prune the element tree, as per
            # http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
            # except that we don't need to prune backlinks from the parent
            # because we don't use LXML.
            # We do this only for <page>s, since we need to inspect the
            # ./revision/text element. The pages comprise the bulk of the
            # file, so in practice we prune away enough.
            elem.clear()


from gensim import utils

from gensim.corpora.wikicorpus import tokenize, remove_markup, remove_template, remove_file, \
    RE_P0, RE_P1, RE_P2, RE_P9, RE_P10, RE_P11, RE_P14, RE_P5, RE_P6, RE_P12, RE_P13


def remove_markup(text, promote_remaining=True, simplify_links=True):
    """Filter out wiki markup from `text`, leaving only text.

    Parameters
    ----------
    text : str
        String containing markup.
    promote_remaining : bool
        Whether uncaught markup should be promoted to plain text.
    simplify_links : bool
        Whether links should be simplified keeping only their description text.

    Returns
    -------
    str
        `text` without markup.

    """
    text = re.sub(RE_P2, '', text)  # remove the last list (=languages)
    # the wiki markup is recursive (markup inside markup etc)
    # instead of writing a recursive grammar, here we deal with that by removing
    # markup in a loop, starting with inner-most expressions and working outwards,
    # for as long as something changes.
    text = remove_template(text)
    text = remove_file(text)
    iters = 0
    while True:
        old, iters = text, iters + 1
        text = re.sub(RE_P0, '', text)  # remove comments
        text = re.sub(RE_P1, '', text)  # remove footnotes
        text = re.sub(RE_P9, '', text)  # remove outside links
        text = re.sub(RE_P10, '', text)  # remove math content
        text = re.sub(RE_P11, '', text)  # remove all remaining tags
        text = re.sub(RE_P14, '', text)  # remove categories
        text = re.sub(RE_P5, '\\3', text)  # remove urls, keep description

        if simplify_links:
            text = re.sub(RE_P6, '\\2', text)  # simplify links, keep description only
        # remove table markup

        text = text.replace('||', '\n|')  # each table cell on a separate line
        text = re.sub(RE_P12, '\n', text)  # remove formatting lines
        text = re.sub(RE_P13, '\n\\3', text)  # leave only cell content
        # remove empty mark-up
        text = text.replace('[]', '')
        # stop if nothing changed between two iterations or after a fixed number of iterations
        if old == text or iters > 2:
            break

    if promote_remaining:
        text = text.replace('[', '').replace(']', '')  # promote all remaining markup to plain text

    return text





def filter_wiki(raw):
    text = utils.to_unicode(raw, 'utf8', errors='ignore')
    text = utils.decode_htmlentities(text)  # '&amp;nbsp;' --> '\xa0'
    return remove_markup(text)#, promote_remaining=False, simplify_links=True)



TOKEN_MIN_LEN = 2
TOKEN_MAX_LEN = 15

import mwparserfromhell
import wikitextparser as wtp

def make_url_from_link(link):
    # url has whitespace at the beginning and end because later on at tokenization it is required that each url gets one token
    return ' http://dbpedia.org/resource/' + link.replace(' ', '_') + ' '


def get_replacement_list(title, wiki_links):
    replacement_map = dict()
    replacement_map[title.lower()] = make_url_from_link(title)
    for link in wiki_links:
        link_text = link.text or link.title
        replacement_map[link_text.lower()] = make_url_from_link(str(link.title))
    return sorted(list(replacement_map.items()), key=lambda x: len(x[0]), reverse=True)



def process_article(article_text, title):
    textone = filter_wiki(article_text)
    parsed_wikicode = mwparserfromhell.parse(article_text)
    text = parsed_wikicode.strip_code()

    text = text.lower()
    for link_text, replacement in get_replacement_list(title, parsed_wikicode.filter_wikilinks()):
     # a lookahead because we dont want to replace "abcdef" when we only have "abc" replace with "123"
        text = re.sub(link_text + '(?![a-zA-Z0-9])', replacement, text)

    #from tokenization.spacytoken import tokenize

    #result = list(tokenize(textone, lowercase=True))

    #text = ' '.join(result)
    #text = text.lower()

    #new_text = []
    #for sentence in tokenize(text, lowercase=True):
    #    bla = ' '.join(sentence)
    #    for link_text, replacement in get_replacement_list(title, parsed_wikicode.filter_wikilinks()):
    #        # a lookahead because we dont want to replace "abcdef" when we only have "abc" replace with "123"
    #        bla = re.sub(link_text + '(?![a-zA-Z0-9])', replacement, bla)
    #        #text = text.replace(link_text, replacement)
    #    new_text.append(bla)


    # text = text.lower()
    # text = text.replace(title.lower(), make_url_from_link(title))
    # for link in parsed_wikicode.filter_wikilinks():
    #     link_text = link.text or link.title
    #     text = text.replace(link_text.lower(),make_url_from_link(str(link.title)))
    #
    #     CONFIGURATION.log(link_text)
        #text.replace('http://dbpedia.org/resource/' + title.replace(' ', '_')))

    #text.replace()



    #text = text.lower()
    #text.replace(title.lower(), 'http://dbpedia.org/resource/' + title.replace(' ', '_'))
    #for link in parsed_wikicode.filter_wikilinks():
    #    text.replace('')
    #CONFIGURATION.log(text)

    #parsed = wtp.parse(article_text)
    #for link in parsed.wikilinks:
    #    CONFIGURATION.log(link.target)
    #    CONFIGURATION.log(link.text)


    #from tokenization.spacytoken import tokenize

    #result = list(tokenize(text))
    #result = tokenize(text, TOKEN_MIN_LEN, TOKEN_MAX_LEN, lower=True)
    #return result
    return None



#from gensim.corpora.wikicorpus import extract_pages

def get_wiki_text():
    with open(source, 'r', encoding='utf-8') as dump_file:
        for title, text, pageid in extract_pages(dump_file, filter_namespaces=set(['0'])):
            pprint.pCONFIGURATION.log((title, pageid))
            process_article(text, title)


if __name__ == '__main__':
    #logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
    logger.info("Start")
    #get_wiki_text()
    import spacy
    nlp = spacy.load("en_core_web_sm") # en or  en_core_web_sm
    doc = nlp(" http://dbpedia.org/resource/Gryffindor  helps is one of the four  http://dbpedia.org/resource/Hogwarts_Houses  of  http://dbpedia.org/resource/Hogwarts_School_of_Witchcraft_and_Wizardry , founded by  http://dbpedia.org/resource/Godric_Gryffindor . godric instructed  http://dbpedia.org/resource/Sorting_Hat  to choose a few particular characteristics he most values. such character traits of students  http://dbpedia.org/resource/Sorting_ceremony  into  http://dbpedia.org/resource/Gryffindor  are courage, chivalry, and determination. the emblematic animal is a  http://dbpedia.org/resource/lion , and its colours are scarlet and gold.  http://dbpedia.org/resource/Nicholas_de_Mimsy-Porpington , also known as \"nearly headless nick\" is the house  http://dbpedia.org/resource/ghost .  ")
    for sentence in doc.sents:
        CONFIGURATION.log(sentence)#[token.string.strip().lower() for token in sentence])