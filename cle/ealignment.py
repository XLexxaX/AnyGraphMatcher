# -*- coding: utf-8 -*-

import gensim
import os
import codecs
import logging
import pandas as pd
import numpy as np


# Log output. Also useful to show program is doing things
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# models trained using gensim implementation of word2vec
print('Loading models...')
modelpath ="D:/Development/Code/melt/examples/simpleJavaMatcher/oaei-resources/AnyGraphMatcher/result_data/cli_task_2019_11_03_19_31_38_909532/w2v.model"
#D:/Development/Code/melt/examples/simpleJavaMatcher/oaei-resources/AnyGraphMatcher/result_data/cli_task_2019_11_01_12_55_35_637929/
model_source = gensim.models.Word2Vec.load(modelpath)
model_target = gensim.models.Word2Vec.load(modelpath)

# list of word pairs to train translation matrix as csv
# eg:
#  source,target
#  今日は、hello
#  犬、dog
#  猫、cat
print('Reading training pairs...')
mappingspath = "C:/Users/allue/oaei_track_cache/tmpdata/baseline.csv"
#word_pairs = codecs.open(mappingspath, 'r', 'utf-8')

pairs = pd.read_csv(mappingspath, encoding="UTF-8", sep="\t", header=None, index_col=False)
pairs.columns = ['source','target']
print(pairs.head(3))

print('Removing missing vocabulary...')

missing = 0

for n in range (len(pairs)):
	if pairs['source'][n] not in model_source.wv.vocab or pairs['target'][n] not in model_target.wv.vocab:
		missing = missing + 1
		pairs = pairs.drop(n)

pairs = pairs.reset_index(drop = True)
print('Amount of missing vocab: ', missing)

# make list of pair words, excluding the missing vocabs
# removed in previous step
pairs['vector_source'] = [model_source.wv[pairs['source'][n]] for n in range (len(pairs))]
pairs['vector_target'] = [model_target.wv[pairs['target'][n]] for n in range (len(pairs))]

# first 5000 from both languages, to train translation matrix
source_training_set = pairs['vector_source'][:5000]
target_training_set = pairs['vector_target'][:5000]

matrix_train_source = pd.DataFrame(source_training_set.tolist()).values
matrix_train_target = pd.DataFrame(target_training_set.tolist()).values

print('Generating translation matrix')

# Matrix W is given in  http://stackoverflow.com/questions/27980159/fit-a-linear-transformation-in-python
translation_matrix = np.linalg.pinv(matrix_train_source).dot(matrix_train_target).T
print('Generated translation matrix')

#print(str(translation_matrix.shape))
#print(str(translation_matrix.dot(model_source.wv["http://dbkwik.webdatacommons.org/marvelcinematicuniverse.wikia.com/resource/Flint"]).tolist()))

pmpath = "C:/Users/allue/oaei_track_cache/tmpdata/possible_matches.csv"
pm = pd.read_csv(pmpath, encoding="UTF-8", sep="\t", header=None, index_col=False)
pm.columns = ['source','target']
writtennids = set()
with open("D:/Development/Code/melt/examples/simpleJavaMatcher/oaei-resources/AnyGraphMatcher/result_data/cli_task_2019_11_03_19_31_38_909532/aligned_embeddings.csv", mode="w+", encoding="UTF-8") as f1:
	f1.write("src_0\tsrc_1\tsrc_2\tsrc_3\tsrc_4\tsrc_5\tsrc_6\tsrc_7\tsrc_8\tsrc_9\tsrc_10\tsrc_11\tsrc_12\tsrc_13\tsrc_14\tsrc_15\tsrc_16\tsrc_17\tsrc_18\tsrc_19\tsrc_20\tsrc_21\tsrc_22\tsrc_23\tsrc_24\tsrc_25\tsrc_26\tsrc_27\tsrc_28\tsrc_29\tsrc_30\tsrc_31\tsrc_32\tsrc_33\tsrc_34\tsrc_35\tsrc_36\tsrc_37\tsrc_38\tsrc_39\tsrc_40\tsrc_41\tsrc_42\tsrc_43\tsrc_44\tsrc_45\tsrc_46\tsrc_47\tsrc_48\tsrc_49\tsrc_50\tsrc_51\tsrc_52\tsrc_53\tsrc_54\tsrc_55\tsrc_56\tsrc_57\tsrc_58\tsrc_59\tsrc_60\tsrc_61\tsrc_62\tsrc_63\tsrc_64\tsrc_65\tsrc_66\tsrc_67\tsrc_68\tsrc_69\tsrc_70\tsrc_71\tsrc_72\tsrc_73\tsrc_74\tsrc_75\tsrc_76\tsrc_77\tsrc_78\tsrc_79\tsrc_80\tsrc_81\tsrc_82\tsrc_83\tsrc_84\tsrc_85\tsrc_86\tsrc_87\tsrc_88\tsrc_89\tsrc_90\tsrc_91\tsrc_92\tsrc_93\tsrc_94\tsrc_95\tsrc_96\tsrc_97\tsrc_98\tsrc_99\tlabel\n")
	for index, row in pm.iterrows():
		if row['source'] not in writtennids:
			f1.write(  "\t".join([str(val) for val in translation_matrix.dot(model_source.wv[row['source']]).tolist()]) + "\t" + row['source'] + "\n")
			writtennids.add(row['source'])
		if row['target'] not in writtennids:
			f1.write(  "\t".join([str(val) for val in model_source.wv[row['target']].tolist()]) + "\t" + row['target'] + "\n")
			writtennids.add(row['target'])


# Returns list of topn closest vectors to vectenter
def most_similar_vector(self, vectenter, topn=5):
    self.init_sims()
    dists = np.dot(self.syn0norm, vectenter)
    if not topn:
        return dists
    best = np.argsort(dists)[::-1][:topn ]
        # ignore (don't return) words from the input
    result = [(self.index2word[sim], float(dists[sim])) for sim in best]
    return result[:topn]

def top_translations(w,numb=5):
    val = most_similar_vector(model_target.wv,translation_matrix.dot(model_source.wv[w]),numb)
    #print('traducwithscofres ', val
    return val


def top_translations_list(w, numb=5):
    val = [top_translations(w,numb)[k][0] for k in range(numb)]
    return val

temp = 1
#top_matches = [ pairs['target'][n] in top_translations_list(pairs['source'][n]) for n in range(5000,5003)]

# print out source word and translation
def display_translations():
    for word_num in range(range_start, range_end):
        source_word =  pairs['source'][word_num]
        translations = top_translations_list(pairs['source'][word_num])
        print (source_word, translations)

# range to use to check accuracy
range_start = 5000
range_end = 6000

#display_translations()

# now we can check for accuracy on words 5000-6000, 1-5000 used to traning
# translation matrix

# returns matrix of true or false, true if translation is accuracy, false if not
# accurate means the first translation (most similiar vector in target language)
# is identical
#accuracy_at_five = [pairs['target'][n] in top_translations_list(pairs['source'][n]) for n in range(range_start, range_end)]
#print('Accuracy @5 is ', sum(accuracy_at_five), '/', len(accuracy_at_five)

#accuracy_at_one = [pairs['target'][n] in top_translations_list(pairs['source'][n],1) for n in range(range_start, range_end)]
#print('Accuracy @1 is ', sum(accuracy_at_one), '/', len(accuracy_at_one)
