"""Usage: downloadseals <tdrs_location> <test_data_collection_name> <test_data_version_number> <base_dir>"""
from docopt import docopt
from urllib.parse import quote_plus
from rdflib import Graph
import logging
import requests
import os

logger = logging.getLogger(__name__)

def __create_location(parts):
    first_part = parts[0].strip()
    resulting_location = first_part if first_part.endswith('/') else first_part + '/'
    for part in parts[1:]:
        resulting_location += part if part.endswith('/') else part + '/'
    return resulting_location


def __head_url(url):
    return requests.head(url).status_code == requests.codes.ok


def download_to_file(url, file_path):
    with open(file_path, 'wb') as f:
        f.write(requests.get(url).content)


def download_seals_datasets(tdrs_location, test_data_collection_name, test_data_version_number, base_dir):
    g = Graph()
    g.parse(__create_location([tdrs_location, 'testdata', 'persistent', quote_plus(test_data_collection_name), quote_plus(test_data_version_number), 'suite']), format='xml')
    result = g.query(
        """SELECT ?suiteItemName
           WHERE {
              ?x <http://www.seals-project.eu/ontologies/SEALSMetadata.owl#hasSuiteItem> ?suiteItem .
              ?suiteItem <http://purl.org/dc/terms/identifier> ?suiteItemName . 
           }
           ORDER BY ?suiteItemName""")

    for suite_item_name in result:
        logger.info('downloading ' + suite_item_name[0])
        base_url_components = [tdrs_location, 'testdata', 'persistent', quote_plus(test_data_collection_name),
                               quote_plus(test_data_version_number),'suite', quote_plus(suite_item_name[0]), 'component']
        src = __create_location(base_url_components + ['source'])
        dst = __create_location(base_url_components + ['target'])
        ref = __create_location(base_url_components + ['reference'])
        if __head_url(src) and __head_url(dst) and __head_url(ref):
            directory_path = os.path.join(base_dir,test_data_collection_name + '_' + test_data_version_number, suite_item_name[0])
            os.makedirs(directory_path, exist_ok=True)
            download_to_file(src, os.path.join(directory_path, 'source.xml'))
            download_to_file(dst, os.path.join(directory_path, 'target.xml'))
            download_to_file(ref, os.path.join(directory_path, 'reference.xml'))

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
    arguments = docopt(__doc__)
    download_seals_datasets(arguments['<tdrs_location>'], arguments['<test_data_collection_name>'], arguments['<test_data_version_number>'], arguments['<base_dir>'])
