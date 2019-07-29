import logging
import os
from collections import defaultdict
from operator import itemgetter

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from nparser import parse, Resource

logger = logging.getLogger(__name__)
package_directory = os.path.dirname(os.path.abspath(__file__))


def create_word2vec_model(path_to_text, cacheFile):
    if os.path.isfile(cacheFile):
        model = Word2Vec.load(cacheFile)
    else:
        model = Word2Vec(LineSentence(path_to_text))
        model.save(cacheFile)
    return model.wv

def load_extracted_information(list_of_paths):

    property_to_domain_range = defaultdict(set)
    for file_path in list_of_paths:
        with open(file_path, 'rb') as file_obj:
            for s, p, o in parse(file_obj):
                if isinstance(o, Resource):
                    prop = p.value.replace('http://dbkwik.webdatacommons.org/harrypotter/resource/', 'http://dbpedia.org/resource/')
                    subj = s.value.replace('http://dbkwik.webdatacommons.org/harrypotter/resource/','http://dbpedia.org/resource/')
                    obj = o.value.replace('http://dbkwik.webdatacommons.org/harrypotter/resource/','http://dbpedia.org/resource/')
                    property_to_domain_range[prop].add((subj, obj))
    property_list = [(property, len(set_of_domain_range), set_of_domain_range) for property, set_of_domain_range in property_to_domain_range.items()]
    property_list = sorted(property_list, key=itemgetter(1), reverse=True)

    #<class 'tuple'>: ('http://dbkwik.webdatacommons.org/harrypotter/property/species',
    #<class 'tuple'>: ('http://dbkwik.webdatacommons.org/harrypotter/property/born',
    #<class 'tuple'>: ('http://dbkwik.webdatacommons.org/harrypotter/property/blood'

    return list(property_to_domain_range['http://dbkwik.webdatacommons.org/harrypotter/property/born'])
    #return list(property_to_domain_range['http://dbkwik.webdatacommons.org/harrypotter/property/blood'])


from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def run_test(source_target, model):
    list_of_source_target = [(source, target) for source, target in source_target if source in model and target in model]

    diff_vector_list = []
    label = []
    for source, target in list_of_source_target:
        diff_vector = model[source] - model[target]
        diff_vector_list.append(diff_vector)
        label.append(target.replace('http://dbpedia.org/resource/', ''))
    test = np.array(diff_vector_list)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(test)

    fig, ax = plt.subplots()
    ax.scatter(principalComponents[:,0], principalComponents[:,1])
    blub = principalComponents.shape[0]
    for i in range(principalComponents.shape[0]):

        ax.annotate(label[i], (principalComponents[i, 0], principalComponents[i, 1]))
    plt.show()


    test = model.most_similar('http://dbpedia.org/resource/Half-blood', topn=50)
    testtwo = model.most_similar('http://dbpedia.org/resource/Harry_Potter', topn=50)

    source, target = list_of_source_target[0]
    source_test, target_test =list_of_source_target[1]
    #result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
    result = model.most_similar(positive=[source_test, target], negative=[source], topn=50)
    CONFIGURATION.log("test")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
    logger.info("Start")
    path_to_text = os.path.join(package_directory, '..', '..', 'data', 'test', 'result.txt')
    cache = os.path.join(package_directory, '..', '..', 'data', 'test', 'word2vec.model')
    extracted_info = os.path.join(package_directory, '..', '..', 'data', 'test', 'infobox-properties.ttl')
    word_model = create_word2vec_model(path_to_text, cache)
    source_target = load_extracted_information([extracted_info])

    run_test(source_target, word_model)