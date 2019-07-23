import logging
import os
import re
import multiprocessing
from wikitext.parsewikitextown import get_raw_text_and_links_from_markup
#from wikitext.parsewikitexthellparser import get_raw_text_and_links_from_markup
from wikitext.processwikidump import extract_pages
from wikitext.wikitokenize import tokenize_spacy

from gensim import utils

logger = logging.getLogger(__name__)
package_directory = os.path.dirname(os.path.abspath(__file__))


def process_link_mapping(title, links):
    processed_map = dict()
    for link_text, link_target in links.items():
        processed_map[link_text.strip().lower()] = 'http://dbpedia.org/resource/' + link_target.strip().replace(' ', '_') + ' '
    # at the end of the url a whitespace to ensure the url ends there
    processed_map[title.strip().lower()] = 'http://dbpedia.org/resource/' + title.strip().replace(' ', '_') + ' '

    replacement_list = sorted(list(processed_map.items()), key=lambda x: len(x[0]), reverse=True)
    return replacement_list

def replace_text(text, links, title):
    text = text.lower()
    replacement_list = process_link_mapping(title, links)
    for link_text, replacement in replacement_list:
        # a lookahead because we dont want to replace "abcdef" when we only have "abc" replace with "123"
        #(?<!\w)dor(?!\w)
        text = re.sub('(?<!\w)' + re.escape(link_text) + '(?!\w)', replacement, text)
    return text

def process_page(args):#title, text, pageid):
    title, text, pageid = args
    print("Process " + title + ' (' + pageid + ')')
    text, links = get_raw_text_and_links_from_markup(text)
    text = replace_text(text, links, title)
    text = re.sub('\s+', ' ', text).strip()
    sentences = tokenize_spacy(text)
    return sentences, title, pageid

def process_wiki_dump(source, target, processes=None):
    if processes is None:
        processes = max(1, multiprocessing.cpu_count() - 1)
    print(processes)

    with open(source, 'r', encoding='utf-8') as dump_file, \
         open(target, 'w', encoding='utf-8') as out_file:

        page_generator = extract_pages(dump_file, filter_namespaces=set(['0']))

        #for title, text, pageid in page_generator:
        #    sentences, title, pageid = process_page(title, text, pageid)
        #    for sentence in sentences:
        #        out_file.write(sentence + '\n')

        with multiprocessing.Pool(processes) as pool:
            for group in utils.chunkize(page_generator, chunksize=10 * processes, maxsize=1):
                for sentences, title, pageid in pool.imap(process_page, group):
                    for sentence in sentences:
                        out_file.write(sentence + '\n')



if __name__ == '__main__':
    #logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
    logger.info("Start")
    source = os.path.join(package_directory, '..', '..', 'data', 'test', 'test.xml')  # 'harrypotter_pages_current.xml')
    target = os.path.join(package_directory, '..', '..', 'data', 'test', 'result.txt')
    process_wiki_dump(source, target)
    #run_word2vec()