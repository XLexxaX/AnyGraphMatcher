
import numpy as np
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import gensim.models.word2vec as w2v
import multiprocessing
import re
import os
import gensim
import pandas as pd
import logging


def stem(CONFIGURATION, sents, ngrams=False):
    
    with open(CONFIGURATION.rundir + "w2v_training_material.csv", mode="w+", encoding=CONFIGURATION.encoding) as f:
        for sent in sents:
            tmp = list()
            for expression in sent:
                if not 'http://' in expression:
                    expression_new = list()
                    for word in expression.split(' '):
                        if ngrams:
                            w = re.sub('[^A-z0-9<>]','', word.lower())
                            w = ps.stem(w)
                            if len(w) > 2:
                                grams = [w[i:i+3] for i in range(len(w)-3+1)]
                                expression_new = expression_new + grams
                            else:
                                expression_new.append(w)
                        else:
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
        f.write("<> <>\n")
    return gensim.models.word2vec.LineSentence(CONFIGURATION.rundir + "w2v_training_material.csv")




def tuplize(sents, CONFIGURATION):
    f = open(CONFIGURATION.rundir + "w2v_formatted_training_material.csv",'w+', encoding=CONFIGURATION.encoding)
    for sentence in sents:
        for j in range(0,len(sentence)):
            for k in range(j+1,len(sentence)):
                f.write(sentence[k] + ',' + sentence[j] + "\n")
    f.close()
    df = pd.read_csv(CONFIGURATION.rundir + "w2v_formatted_training_material.csv", sep=',', header=None,
                    encoding=CONFIGURATION.encoding)
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

def prepare_training_data(sentences, CONFIGURATION, ngrams=False):


    sentences = stem(CONFIGURATION, sentences, ngrams)




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

def file_len(fname, CONFIGURATION):
    i=0
    with open(fname, mode="r", encoding=CONFIGURATION.encoding) as f:
        for line in f:
            i=i+1
    return i

def embed(sentences, dim, CONFIGURATION, ngrams = False, window=100):



    logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
    logging.root.level = logging.INFO

    sentences = prepare_training_data(sentences, CONFIGURATION, ngrams)

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
    CONFIGURATION.log(str(multiprocessing.cpu_count()))
    num_workers = int(multiprocessing.cpu_count()*0.75)

    # Context window length.
    context_size = window

    # Downsample setting for frequent words.
    #0 - 1e-5 is good for this
    downsampling = 0.05

    # Seed for the RNG, to make the results reproducible.
    #random number generator
    #deterministic, good for debugging
    seed = 1
    model = w2v.Word2Vec(
        sg=0,
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        negative=0,
        ns_exponent=0.1
    )

    total_examples = file_len(CONFIGURATION.rundir + "w2v_training_material.csv", CONFIGURATION)

    model.build_vocab(sentences)

    epochs = int(-0.237*(os.path.getsize(CONFIGURATION.rundir + "w2v_training_material.csv")/(10**6))+300.0) #int(((os.path.getsize(CONFIGURATION.rundir + "w2v_training_material.csv")/(10**6))**(-2))*675000)
    #epochs = int(epochs/2)
    epochs = max(epochs, 50)
    epochs = min(epochs, 500)
    epochs = 1

    CONFIGURATION.log("      --> Training embeddings with " + str(epochs) + " epochs: 0% [inactive]", end="\r")
    model.train(sentences, total_examples=total_examples, epochs=epochs)
    model.save(CONFIGURATION.rundir+"w2v.model")
    CONFIGURATION.log("      --> Training embeddings with " + str(epochs) + " epochs: 100% [inactive]")
    return model
