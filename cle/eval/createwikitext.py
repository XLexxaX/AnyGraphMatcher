import logging
import lzma
import os
import re

from xml.etree.cElementTree import iterparse  # LXML isn't faster, so let's go with the built-in solution

logger = logging.getLogger(__name__)
package_directory = os.path.dirname(os.path.abspath(__file__))

# https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/scripts/make_wikicorpus.py
# https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/corpora/wikicorpus.py
# https://radimrehurek.com/gensim/corpora/wikicorpus.html


def get_namespace(tag):
    m = re.match("^{(.*?)}", tag)
    namespace = m.group(1) if m else ""
    if not namespace.startswith("http://www.mediawiki.org/xml/export-"):
        raise ValueError("%s not recognized as MediaWiki dump namespace" % namespace)
    return namespace


def extract_pages(f, filter_namespaces=False, filter_articles=None):
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
            title = elem.find(title_path).text
            text = elem.find(text_path).text

            if filter_namespaces:
                ns = elem.find(ns_path).text
                if ns not in filter_namespaces:
                    text = None

            if filter_articles is not None:
                if not filter_articles(
                        elem, namespace=namespace, title=title,
                        text=text, page_tag=page_tag,
                        text_path=text_path, title_path=title_path,
                        ns_path=ns_path, pageid_path=pageid_path):
                    text = None

            pageid = elem.find(pageid_path).text
            yield title, text or "", pageid  # empty page will yield None

            # Prune the element tree, as per
            # http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
            # except that we don't need to prune backlinks from the parent
            # because we don't use LXML.
            # We do this only for <page>s, since we need to inspect the
            # ./revision/text element. The pages comprise the bulk of the
            # file, so in practice we prune away enough.
            elem.clear()


from lxml import etree

def extract_pages_new(f, filter_namespaces):
    # https://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    context = etree.iterparse(f, events=('end',), tag='{*}page') # TODO: or specify the correct one

    for _, elem in context:
        title, ns, id, text = '', '', '', ''
        for child in elem.iterchildren():
            if child.tag.endswith('title'):
                title = child.text
            elif child.tag.endswith('ns'):
                ns = child.text
            elif child.tag.endswith('id'):
                id = child.text
            elif child.tag.endswith('revision'):
                for sub_child in child.iterchildren():
                    if sub_child.tag.endswith('text'):
                        text = sub_child.text
                        break

        yield title, ns, id, text

        # It's safe to call clear() here because no descendants will be accessed
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context


def extract_pages_new_new(f, filter_namespaces):
    # https://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    context = etree.iterparse(f, events=('end',), tag='{*}page') # TODO: or specify the correct one
    title_xpath = etree.XPath("./{http://www.mediawiki.org/xml/export-0.10/}title")

    for _, elem in context:
        title, ns, id, text = '', '', '', ''
        title = title_xpath(elem)

        yield title, ns, id, text

        # It's safe to call clear() here because no descendants will be accessed
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context



def create_wiki_text(file_path_in, file_path_out):
    pass
    #with lzma.open(file_path_in) as f, open(file_path_out, 'w'):
    #    f.extractall(r"<output path>")

    #with open(file_path_in) as f, open(file_path_out, 'w'):



def parsefile():
    source = os.path.join(package_directory, '..', '..', 'data', 'test', 'harrypotter_pages_current.xml')
    with open(source, encoding='utf-8') as f:
        for title, text, page_id in extract_pages(f):
            pass


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
    logger.info("Start")

    #source = os.path.join(package_directory, '..', '..', 'data', 'test', 'test.xml')
    #source = os.path.join(package_directory, '..', '..', 'data', 'test', 'harrypotter_pages_current.xml')
    #target = os.path.join(package_directory, '..', '..', 'data', 'test', 'raw_text.txt')
    #with open(source, 'rb') as f:
    #    #create_wiki_text(source, target)
    #    for title, ns, text, page_id in extract_pages_new_new(f, None):
    #        #CONFIGURATION.log(title)
    #        pass

    #segment_and_write_all_articles

        
        
#https://stackoverflow.com/questions/43078980/python-multiprocessing-with-generator
#https://stackoverflow.com/questions/44708312/how-to-use-a-generator-as-an-iterable-with-multiprocessing-map-function
#https://github.com/rasbt/mputil/blob/master/mputil/map.py
# python pridcer consumer