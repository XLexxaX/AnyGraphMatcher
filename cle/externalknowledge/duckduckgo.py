import requests
import logging
from lxml import html
import urllib
import os
from bs4 import BeautifulSoup
#from boilerpipe.extract import Extractor
#from goose3 import Goose # pip install goose3
import justext # pip install justext
from tokenization.nltktoken import tokenize

from slugify import slugify #pip install  python-slugify

logger = logging.getLogger(__name__)


def search_duckduckgo(keywords, max_results=None, timeout=None):
    """Always fetches 30 links at once and yields it one after the other."""
    url = 'https://duckduckgo.com/html/'
    params = {
        'q': keywords,
        's': '0',
    }
    yielded = 0
    while True:
        doc = html.fromstring(requests.post(url, data=params, timeout=timeout).text)
        links = doc.xpath('//div[@id="links"]/div[contains(@class, "web-result")]/div[contains(@class, "links_main")]/a/@href')
        for link in links:
            yield link
            yielded += 1
            if max_results and yielded >= max_results:
                return
        try:
            params = dict(doc.xpath('//div[@id="links"]/div[contains(@class, "nav-link")]/form')[0].form_values())
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            return



def html_to_text(html_content_in_byte):
    soup = BeautifulSoup(html_content_in_byte, 'lxml')
    return soup.get_text(' ', strip=True)

#def html_to_text_boilerpipe(link):
#    extractor = Extractor(extractor='ArticleExtractor', url=link)
#    CONFIGURATION.log(extractor.getText())

#def html_to_text_goose(link):
#    g = Goose()
#    article = g.extract(url=link)
#    CONFIGURATION.log(articleaned_text)

def html_to_text_justext(html_content_in_byte):
    paragraphs = justext.justext(html_content_in_byte, justext.get_stoplist("English"))
    boilerplate_free = [paragraph.text for paragraph in paragraphs if not paragraph.is_boilerplate]
    return "".join(boilerplate_free)


def generate_text(query, amount_links=30, timeout=2):
    for link in search_duckduckgo(query, amount_links, timeout=timeout):
        logger.debug("Check link: %s", link)
        try:
            response = requests.get(link, timeout=timeout)
        except requests.exceptions.Timeout:
            logger.debug("Link %s - time out, continue", link)
            continue
        except requests.exceptions.RequestException:
            logger.debug("Link %s - RequestException, continue", link)
            continue
        text = html_to_text_justext(response.content)
        for sent in tokenize(text):
            yield sent


def save_all_html(folder_path, query, amount_links=30, timeout=1):
    for link in search_duckduckgo(query, amount_links, timeout=timeout):
        response = requests.get(link, timeout=timeout)
        logger.debug("Check link: %s", link)
        with open(os.path.join(folder_path, slugify(link)), 'w', encoding='utf-8') as f:
            #text = html_to_text_justext(response.content)
            text = html_to_text(response.content)
            for sent in tokenize(text):
                f.write(" ".join(sent) + '\n')

#boilerplate removal:
#http://ws-dl.blogspot.com/2017/03/2017-03-20-survey-of-5-boilerplate.html
#goose:
#https://pypi.org/search/?q=goose
#boilerpipe:
#https://github.com/kohlschutter/boilerpipe
#https://github.com/misja/python-boilerpipe
#python-boilerpipe
#jusText
#https://github.com/miso-belica/jusText
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)
    logging.info("Start")

    #search_duckduckgo('my conference', max_results=10)

    #link_to_text_justext('https://en.wikipedia.org/wiki/Peer_review')

    #for link in search_duckduckgo('peer review', max_results=10):
    #    logger.info(link)

    save_all_html(r'C:\Users\shertlin\Desktop\bal', 'adipose tissue')