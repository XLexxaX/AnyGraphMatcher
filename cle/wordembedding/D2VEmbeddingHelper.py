
import numpy as np
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
import re
import os
import gensim
import pandas as pd

def stem(CONFIGURATION, sents):

    with open(CONFIGURATION.rundir + "w2v_training_material.csv", mode="w+", encoding="UTF-8") as f:
        for sent in sents:
            tmp = list()
            for expression in sent:
                if not 'http://' in expression:
                    expression_new = list()
                    for word in expression.split(' '):
                        expression_new.append(ps.stem(re.sub('[^A-z0-9<>]','', word.lower())))
                    expression = expression_new
                else:
                    expression = [re.sub('[\r\n]', '', expression)]
                if not " ".join(expression) == '' and len(" ".join(expression))>1:
                    tmp = tmp + expression
            if len(tmp) > 1:
                for x in tmp:
                    f.write(str(x) + " ")
                f.write("\n")
        #f.write("<> <>\n")

    from gensim.test.utils import datapath
    from gensim.models.doc2vec import TaggedLineDocument
    for document in TaggedLineDocument(datapath(CONFIGURATION.rundir + "w2v_training_material.csv")):
        yield document




def tuplize(sents, CONFIGURATION):
    f = open(CONFIGURATION.rundir + "w2v_formatted_training_material.csv",'w+', encoding="UTF-8")
    for sentence in sents:
        for j in range(0,len(sentence)):
            for k in range(j+1,len(sentence)):
                f.write(sentence[k] + ',' + sentence[j] + "\n")
    f.close()
    df = pd.read_csv(CONFIGURATION.rundir + "w2v_formatted_training_material.csv", sep=',', header=None,
                    encoding="UTF-8")
    return df

def eliminate_rare_and_freqeunt_terms(x):
    t = x.groupby(0).agg(['count'])[1].sum()['count']
    MAXOCCURENCE = 0.3 * t
    ctr = 0
    sort = x.groupby(0).agg(['count'])[1].sort_values(by=['count'], ascending=True)
    for ind, v in sort.iterrows():
        ctr = ctr + v['count']
        if ctr > MAXOCCURENCE:
            MAXCOUNT = v['count']
            break
    MINOCCURENCE = 0.001 * t
    ctr = 0
    sort = x.groupby(0).agg(['count'])[1].sort_values(by=['count'], ascending=False)
    for ind, v in sort.iterrows():
        ctr = ctr + v['count']
        if ctr > MAXOCCURENCE:
            MINCOUNT = v['count']
            break

    MINCOUNT = 1
    MAXCOUNT = 999999
    y = x.groupby(0).agg(['count'])[1] > MINCOUNT
    y = y.loc[y['count'] == True]
    z = x.groupby(0).agg(['count'])[1] > MAXCOUNT
    z = z.loc[z['count'] == True]
    x = x.loc[x[0].isin(list(y.index) + list(z.index))]

    return x

def prepare_training_data(sentences, CONFIGURATION):


    #sentences = stem(CONFIGURATION, sentences)
    ctr = 0
    with open(CONFIGURATION.rundir + "w2v_training_material.csv", mode="w+", encoding="UTF-8") as f:
        for sent in sentences:
            tmp = list()
            for expression in sent:
                if not 'http://' in expression:
                    expression_new = list()
                    for word in expression.split(' '):
                        word = (ps.stem(re.sub('[^A-z0-9<>]','', word.lower())))
                        #if len(word)>2:
                        #    words = #[word[i:i+3] for i in range(len(word)-3+1)]
                        #else:
                        words = [word]
                        expression_new = expression_new + words
                    expression = expression_new
                else:
                    expression = [re.sub('[\r\n]', '', expression)]
                #if not " ".join(expression) == '' and len(" ".join(expression))>1:
                tmp = tmp + expression
            if len(tmp) > 0:
                for x in tmp:
                    f.write(str(x) + " ")
                f.write("\n")
                ctr += 1
        #f.write("<> <>\n")

    from gensim.test.utils import datapath
    from gensim.models.doc2vec import TaggedLineDocument
    sentences = [document for document in TaggedLineDocument(datapath(CONFIGURATION.rundir + "w2v_training_material.csv"))]




    #x = tuplize(sentences, CONFIGURATION)

    #x = eliminate_rare_and_frequent_terms(x)



    #documents = list()
    #for index, row in x.iterrows():
    #    documents.append([str(row[0])] + [str(row[1])])

    #documents = literalize(documents)

    documents = sentences

    return documents

def literalize(documents):
    d2 = list()
    for d in documents:
        for dx in d:
            if not 'http' in dx:
                d2.append(d)
                break
    return d2

def embed(sentences, dim, CONFIGURATION, window=100):


    sentences = prepare_training_data(sentences, CONFIGURATION)

    import gensim.models.word2vec as w2v
    import multiprocessing
    #ONCE we have vectors
    #step 3 - build model
    #3 main tasks that vectors help with
    #DISTANCE, SIMILARITY, RANKING

    # Dimensionality of the resulting word vectors.
    #more dimensions, more computationally expensive to train
    #but also more accurate
    #more dimensions = more generalized
    num_features = dim
    # Minimum word count threshold.
    min_word_count = 1

    # Number of threads to run in parallel.
    #more workers, faster we train
    num_workers = multiprocessing.cpu_count()*2

    # Context window length.
    context_size = window


    model = Doc2Vec(
        vector_size=num_features,
        min_count=min_word_count,
        epochs=500,
        negative=20,
        ns_exponent=0.1,
        workers=num_workers)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model.save(CONFIGURATION.rundir+"w2v.model")
    return model
