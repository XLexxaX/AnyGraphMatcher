import pandas as pd
import numpy as np
import ntpath

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from pandas_ml import ConfusionMatrix

from configurations.PipelineTools import PipelineDataTuple
from matcher import PredictionToXMLConverter
from visualization import EmbeddingSaver
import sys
import os
from joblib import dump, load
from sklearn.metrics.pairwise import *
import re
from gensim.models import Doc2Vec, Word2Vec


global CONFIGURATION

def exec(graph1, graph2):



    EmbeddingSaver.interface(PipelineDataTuple(graph1, graph2), None, CONFIGURATION)

    basedir = CONFIGURATION.rundir

    gs = pd.read_csv(CONFIGURATION.gold_mapping.raw_testsets[0], encoding="UTF-8", sep="\t", header=None)
    gs.columns = ['src_id','tgt_id']
    embs = pd.read_csv(basedir+"stratified_embeddings.csv", encoding="UTF-8", sep=",")
    embs = embs[[col for col in embs.columns if re.match('x\d+', col) is not None]+['label']]
    embs.columns = ["src_" + str(col) for col in [re.search("\d+", col).group(0) for col in embs.columns if re.match('x\d+', col) is not None]] + ['label']
    gs = gs.merge(embs, left_on=['src_id'], right_on=['label'])
    embs.columns = ["tgt_" + str(col) for col in [re.search("\d+", col).group(0) for col in embs.columns if re.match('src_\d+', col) is not None]] + ['label']
    gs = gs.merge(embs, left_on=['tgt_id'], right_on=['label'])


    print("1")
    labels1 = dict()
    categories1 = dict()
    with open(CONFIGURATION.src_triples, encoding="UTF-8", mode="r") as f:
        for line in f:
            if " <" + CONFIGURATION.properties.src_label_properties[0] + "> " in line:
                line = line.replace("<","").replace(">","").replace(" .\n","").split(" "+CONFIGURATION.properties.src_label_properties[0]+" ")
                labels1[line[0]] = line[1]
            if " <" + CONFIGURATION.properties.category_property + "> " in line:
                line = line.replace("<","").replace(">","").replace(" .\n","").split(" "+CONFIGURATION.properties.category_property+" ")
                if line[1] not in [CONFIGURATION.properties.class_descriptor,CONFIGURATION.properties.property_descriptor]:
                    categories1[line[0]] = 'resource'
                else:
                     categories1[line[0]] = line[1]
    labels2 = dict()
    categories2 = dict()
    with open(CONFIGURATION.tgt_triples, encoding="UTF-8", mode="r") as f:
                for line in f:
                    if " <" + CONFIGURATION.properties.tgt_label_properties[0] + "> " in line:
                        line = line.replace("<", "").replace(">", "").replace(" .\n", "").split(
                            " " + CONFIGURATION.properties.tgt_label_properties[0] + " ")
                        labels2[line[0]] = line[1]
                    if " <" + CONFIGURATION.properties.category_property + "> " in line:
                        line = line.replace("<", "").replace(">", "").replace(" .\n", "").split(
                            " " + CONFIGURATION.properties.category_property + " ")
                        if line[1] not in [CONFIGURATION.properties.class_descriptor,
                                           CONFIGURATION.properties.property_descriptor]:
                            categories2[line[0]] = 'resource'
                        else:
                            categories2[line[0]] = line[1]

    # In[288]:
    print("2")

    def get_category(categoriesdict, labels):
        cats = list()
        keys = categoriesdict.keys()
        for l in labels:
            if l in keys:
                cats.append(categoriesdict[l])
            else:
                cats.append("resource")
        return cats


    gs.loc[:, 'src_category'] = get_category(categories1, gs['src_id'].tolist())
    gs.loc[:, 'tgt_category'] = get_category(categories2, gs['tgt_id'].tolist())
    gs = gs.loc[gs.src_category == gs.tgt_category]
    len(gs)
    print("3")
    # In[308]:


    def extend_features(df):
        src_pattern = "src_\d+"
        tgt_pattern = "tgt_\d+"
        src_dim = int(len([elem for elem in [re.match(src_pattern, elem) is not None for elem in df.columns.values.tolist()] if elem==True]))
        tgt_dim = int(len([elem for elem in [re.match(tgt_pattern, elem) is not None for elem in df.columns.values.tolist()] if elem==True]))


        def dotproduct(v1, v2):
            result = list()
            for i in range(len(v1)):
                result.append([np.dot(v1[i], v2[i])])
            return np.array(result)

        def length(v):
            return np.sqrt(dotproduct(v, v))

        def angle(v1, v2):
            return np.arctan(dotproduct(v1, v2) / (length(v1) * length(v2)))

        a = np.array(df[["src_" + str(i) for i in range(src_dim)]].values.tolist())
        b = np.array(df[["tgt_" + str(i) for i in range(tgt_dim)]].values.tolist())
        df['src_tgt_angle'] = paired_cosine_distances(a, b)
        src_origin = np.full((len(df), src_dim), 0.0000001)
        tgt_origin = np.full((len(df), tgt_dim), 0.0000001)
        df['src_angle_to_origin'] = paired_cosine_distances(tgt_origin,a)
        df['tgt_angle_to_origin'] = paired_cosine_distances(src_origin,b)
        df['src_veclen'] = length(a)
        df['tgt_veclen'] = length(b)
        df['src_tgt_veclen'] = paired_euclidean_distances(a,b)#.diagonal()#length(a-b)
        df.head()

        df.fillna(0, inplace = True)
        return df


    # In[322]:


    gs = extend_features(gs)
    #oaei_gold_standard3 = extend_features(oaei_gold_standard3)


    # In[310]:


    memo = {}

    def jacc(s,t, n=3):
        s = labels1[s]
        t = labels2[t]
        t = set([t[i:i+n] for i in range(len(t)-n+1)])
        s = set([s[i:i+n] for i in range(len(s)-n+1)])
        return 1-len([gram for gram in s if gram in t])/max(len(s), len(t))

    def lev(s,t, n=3):
        s = labels1[s]
        t = labels2[t]
        return levenshtein(s,t)/max(len(s),len(t))

    def levenshtein(s, t):
        if s == "":
            return len(t)
        if t == "":
            return len(s)
        cost = 0 if s[-1] == t[-1] else 1

        i1 = (s[:-1], t)
        if not i1 in memo:
            memo[i1] = levenshtein(*i1)
        i2 = (s, t[:-1])
        if not i2 in memo:
            memo[i2] = levenshtein(*i2)
        i3 = (s[:-1], t[:-1])
        if not i3 in memo:
            memo[i3] = levenshtein(*i3)
        res = min([memo[i1]+1, memo[i2]+1, memo[i3]+cost])

        return res
    gs['syntactic_diff'] = gs.apply(lambda row: jacc(row['src_id'], row['tgt_id']), axis=1)
    gs['plus_diff'] = gs.apply(lambda row: lev(row['src_id'], row['tgt_id']), axis=1)


    print("4")

    #matchings = gs.loc[gs.plus_diff<0.25]
    #gs = matchings
    #len(matchings)


    # In[260]:



    model = Word2Vec.load(basedir+"w2v.model")
    #def get_training_material(nid):
    #            res = list()
    #            with open(basedir+"w2v_training_material.csv", mode="r", encoding="UTF-8") as f:
    #                for line in f:
    #                    if nodeid in line.split(" "):
    #                        for w in line.split(" "):
    #                            yield w


    # In[326]:


    gs.loc[:,'total_score'] = 0


    # In[327]:


    progress = 0
    matchings = list()
    total = len(set(gs.src_id))
    for nodeid in set(gs.src_id):
                    x = gs.loc[(gs.src_id==nodeid) ]

                    progress += 1
                    if len(x)<1:
                        continue
                    print(str(int(100*progress/total)) + "% done")


                    ctr = 1
                    x = x.sort_values(by=['src_tgt_angle'], ascending=True)
                    x.loc[:,'cos_score'] = 0
                    for index, row in x.iterrows():
                        x.loc[index, 'cos_score'] = row['cos_score'] + 1/ctr
                        ctr += 1


                    #x.loc[:,'conficence'] = 0
                    x.loc[:,'confidence_score'] = 0
                    #ctr = 1
                    #for index, row in x.iterrows():
                    #    x.loc[index, 'confidence_score'] = row['confidence_score'] + 1/ctr
                    #    ctr += 1

                    x = x.sort_values(by=['src_tgt_veclen'], ascending=True)
                    x.loc[:,'euclid_score'] = 0
                    ctr = 1
                    ##x.columns = ['euclid_sim' if col==0 else col for col in x.columns]
                    for index, row in x.iterrows():
                        #print(row[1] + " - " + str(row['euclid_sim']))
                        x.loc[index, 'euclid_score'] = row['euclid_score'] + 1/ctr
                        ctr += 1

                    x = x.sort_values(by=['plus_diff'], ascending=True)
                    x.loc[:,'syntax_score'] = 0
                    ctr = 1
                    ##x.columns = ['euclid_sim' if col==0 else col for col in x.columns]
                    for index, row in x.iterrows():
                        #print(row[1] + " - " + str(row['euclid_sim']))
                        x.loc[index, 'syntax_score'] = row['syntax_score'] + 1/ctr
                        ctr += 1


                    #o1 = get_training_material(nodeid)
                    #inverted_dict = dict()
                    #for tgt_id in x.tgt_id.to_list():
                    #    inverted_dict[tgt_id] = [tgt_id]
                    #    for word in get_training_material(tgt_id):
                    #        if word in inverted_dict.keys():
                    #            inverted_dict[word].append(tgt_id)
                    #        else:
                    #            inverted_dict[word] = [tgt_id]

                    #tgt_scores = dict()
                    #for tgt_id in x.tgt_id.to_list():
                    #    tgt_scores[tgt_id]=0

                    #for tupl in model.predict_output_word(o1, topn=99999999999999):
                    #    if tupl[0] in inverted_dict.keys():
                    #        for tgt_id in inverted_dict[tupl[0]]:
                    #            tgt_scores[tgt_id] = tgt_scores[tgt_id] + tupl[1]
                    x.loc[:,'probability'] = 0
                    x.loc[:,'probability_score'] = 0
                    #ctr=1
                    #for item in sorted(tgt_scores.items(), key = lambda kv:(kv[1], kv[0]), reverse=True):
                    #    x.loc[x.tgt_id==item[0],'probability_score'] = 1/ctr
                    #    x.loc[x.tgt_id==item[0],'probability'] = float(item[1])
                    #    ctr += 1

                    # In[316]:


                    #x.loc[:,'total_score'] = x['syntax_score'] + x['euclid_score'] + x['probability_score'] + x['confidence_score'] + x['cos_score']
                    #x = x.sort_values(by=['total_score'], ascending=False)
                    ##x.columns = ['tgt_id' if col==1 else col for col in x.columns]

                    #for index, row in x.iterrows():#x.loc[x.total_score == max(x.total_score.values),:].iterrows():
                    #    matching_pair = pd.DataFrame([x.loc[index]])
                    #    matching_pair.loc[:,'src_id'] = nodeid
                    #    #print(nodeid + "\t" + row[1] + "\t" + str(row['total_score']) + "\t" + str(row['cos_score']) + "\t" + str(row['euclid_score']))
                    matchings.append(x)

    matchings = pd.concat(matchings, axis=0)
    gs = matchings
    matchings = list()
    progress = 0
    for nodeid in set(gs.tgt_id):
                x = gs.loc[(gs.tgt_id==nodeid) ]

                progress += 1
                if len(x)<1:
                    continue
                print(str(int(100*progress/total)) + "% done")



                ctr = 1
                x = x.sort_values(by=['src_tgt_angle'], ascending=True)
                #x.loc[:,'cos_score'] = 0
                for index, row in x.iterrows():
                    x.loc[index, 'cos_score'] = row['cos_score'] + 1/ctr
                    ctr += 1


                #x.loc[:,'conficence'] = 0
                #x.loc[:,'confidence_score'] = 0
                #ctr = 1
                #for index, row in x.iterrows():
                #    x.loc[index, 'confidence_score'] = row['confidence_score'] + 1/ctr
                #    ctr += 1

                x = x.sort_values(by=['src_tgt_veclen'], ascending=True)
                #x.loc[:,'euclid_score'] = 0
                ctr = 1
                ##x.columns = ['euclid_sim' if col==0 else col for col in x.columns]
                for index, row in x.iterrows():
                    #print(row[1] + " - " + str(row['euclid_sim']))
                    x.loc[index, 'euclid_score'] = row['euclid_score'] + 1/ctr
                    ctr += 1

                x = x.sort_values(by=['plus_diff'], ascending=True)
                #x.loc[:,'syntax_score'] = 0
                ctr = 1
                ##x.columns = ['euclid_sim' if col==0 else col for col in x.columns]
                for index, row in x.iterrows():
                    #print(row[1] + " - " + str(row['euclid_sim']))
                    x.loc[index, 'syntax_score'] = row['syntax_score'] + 1/ctr
                    ctr += 1


                #o1 = get_training_material(nodeid)
                #inverted_dict = dict()
                #for tgt_id in x.tgt_id.to_list():
                #    inverted_dict[tgt_id] = [tgt_id]
                #    for word in get_training_material(tgt_id):
                #        if word in inverted_dict.keys():
                #            inverted_dict[word].append(tgt_id)
                #        else:
                #            inverted_dict[word] = [tgt_id]

                #tgt_scores = dict()
                #for tgt_id in x.tgt_id.to_list():
                #    tgt_scores[tgt_id]=0

                #for tupl in model.predict_output_word(o1, topn=99999999999999):
                #    if tupl[0] in inverted_dict.keys():
                #        for tgt_id in inverted_dict[tupl[0]]:
                #            tgt_scores[tgt_id] = tgt_scores[tgt_id] + tupl[1]
                #x.loc[:,'probability'] = 0
                #x.loc[:,'probability_score'] = 0
                #ctr=1
                #for item in sorted(tgt_scores.items(), key = lambda kv:(kv[1], kv[0]), reverse=True):
                #    x.loc[x.tgt_id==item[0],'probability_score'] = 1/ctr
                #    x.loc[x.tgt_id==item[0],'probability'] = float(item[1])
                #    ctr += 1

                # In[316]:


                x.loc[:,'total_score'] = x['syntax_score'] + x['euclid_score'] + x['probability_score'] + x['confidence_score'] + x['cos_score']
                x = x.sort_values(by=['total_score'], ascending=False)
                for index, row in x.iterrows():
                    matching_pair = pd.DataFrame([x.loc[index]])
                    matchings.append(matching_pair)

    matchings = pd.concat(matchings, axis=0)

    # In[263]:


    #matchings = matchings_saved
    #matchings = gs


    # In[264]:


    matchings = matchings.sort_values(by=['total_score','src_tgt_angle'], ascending=[False, True])
    married_matchings = list()
    ctr = 0
    while len(matchings) > 0:
                    ctr += 1
                    row = matchings.head(1)
                    married_matchings.append(row)
                    matchings = matchings.loc[~(matchings.src_id == row.src_id.values[0]) & ~(matchings.tgt_id == row.tgt_id.values[0])]
                    print(str(len(matchings)) + " left     ", end="\r")
    married_matchings = pd.concat(married_matchings, axis=0)

    married_matchings[['src_id','tgt_id']].to_csv(basedir+"married_matchings.csv", encoding="UTF-8", sep="\t")
    PredictionToXMLConverter.interface(PipelineDataTuple(graph1, graph2), PipelineDataTuple('married_matchings.csv'), CONFIGURATION)




    return PipelineDataTuple(graph1, graph2)


def interface(main_input, args, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    graph1 = main_input.get(0)
    graph2 = main_input.get(1)
    assert graph1 is not None, "Graph (1) not found in " + os.path.basename(sys.argv[0])
    assert graph2 is not None, "Graph (2) not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.logfile is not None, "Path to logfile not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.name is not None, "Test config name not found in " + os.path.basename(sys.argv[0])
    return exec(graph1, graph2)
