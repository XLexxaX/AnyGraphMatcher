from sentencegenerator.saveutil import save_sentences
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from collections import defaultdict
from tokenization.spacytoken import tokenize

__default_set_of_properties = set([
    'http://www.w3.org/2000/01/rdf-schema#label',
    'http://www.w3.org/2000/01/rdf-schema#comment',
    'http://purl.org/dc/terms/identifier',
    'http://dbkwik.webdatacommons.org/ontology/abstract'
])

def __make_flat_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def __get_tagged_documents(kg, properties, with_fragments):
    text_per_resource = defaultdict(set)
    for s, p, o in kg.get_literal_triples_with_fragments() if with_fragments else kg.get_literal_triples():
        if p in properties:
            for sent in tokenize(o):
                text_per_resource[s].add(tuple(sent))#update(tuple(tokenize(o)))
    return [TaggedDocument(__make_flat_list(sentences), [uri]) for uri, sentences in text_per_resource.items()]


def doc2vec_embedding(kg, properties=__default_set_of_properties, with_fragments=True, dm=1, vector_size=100, window=5, epochs=30):
    list_of_tagged_documents = __get_tagged_documents(kg, properties, with_fragments)

    doc2vec_model = Doc2Vec(documents=list_of_tagged_documents,
                           dm=dm, vector_size=vector_size, window=window, epochs=epochs, sample=0, min_count=0)

    return doc2vec_model.docvecs


def doc2vec_embedding_from_kg(properties=__default_set_of_properties, with_fragments=True, dm=1, vector_size=100, window=5, epochs=30):
    return lambda kg: doc2vec_embedding(kg, properties, with_fragments, dm, vector_size, window, epochs)
