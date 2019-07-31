import pandas as pd
import numpy as np
import ntpath

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from pandas_ml import ConfusionMatrix

from configurations.PipelineTools import PipelineDataTuple
from matcher import PredictionToXMLConverter, RAMCleaner
from visualization import EmbeddingSaver
import sys
import os
from xgboost import XGBClassifier
from joblib import dump, load
from sklearn.metrics.pairwise import *
import re
from gensim.models import Doc2Vec, Word2Vec


global CONFIGURATION

def exec(graph1, graph2):

    match(prepare(graph1, graph2), graph1, graph2)
    return PipelineDataTuple(graph1, graph2)


def prepare(graph1, graph2):
        EmbeddingSaver.interface(PipelineDataTuple(graph1, graph2), None, CONFIGURATION)
        CONFIGURATION.log("      --> Cleaning memory: 0% [inactive]", end="\r")
        RAMCleaner.interface(PipelineDataTuple(graph1, graph2), None, CONFIGURATION)

        CONFIGURATION.log("      --> Cleaning memory: 100% [inactive]")

        basedir = CONFIGURATION.rundir

        


        pm = pd.read_csv(CONFIGURATION.gold_mapping.raw_testsets[0], encoding="UTF-8", sep="\t", header=None)
        pm.columns = ['src_id','tgt_id']
        embs = pd.read_csv(basedir+"stratified_embeddings.csv", encoding="UTF-8", sep="\t")
        #embs = embs[[col for col in embs.columns if re.match('x\d+', col) is not None]+['label']]
        #embs.columns = ["src_" + str(col) for col in [re.search("\d+", col).group(0) for col in embs.columns if re.match('src_\d+', col) is not None]] + ['label']
        pm = pm.merge(embs, left_on=['src_id'], right_on=['label'])
        embs.columns = ["tgt_" + str(col) for col in [re.search("\d+", col).group(0) for col in embs.columns if
                                                      re.match('src_\d+', col) is not None]] + ['label']
        pm = pm.merge(embs, left_on=['tgt_id'], right_on=['label'])
        


        gs = pd.read_csv(CONFIGURATION.gold_mapping.raw_trainsets[0], encoding="UTF-8", sep="\t", header=None)
        gs.columns = ['src_id','tgt_id','target']
        embs.columns = ["src_" + str(col) for col in [re.search("\d+", col).group(0) for col in embs.columns if re.match('tgt_\d+', col) is not None]] + ['label']
        gs = gs.merge(embs, left_on=['src_id'], right_on=['label'])
        embs.columns = ["tgt_" + str(col) for col in [re.search("\d+", col).group(0) for col in embs.columns if
                                                      re.match('src_\d+', col) is not None]] + ['label']
        gs = gs.merge(embs, left_on=['tgt_id'], right_on=['label'])


        CONFIGURATION.log("      --> Applying ontology restrictions: 0% [inactive]", end="\r")

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

        def get_category(categoriesdict, labels):
            cats = list()
            keys = categoriesdict.keys()
            for l in labels:
                if l in keys:
                    cats.append(categoriesdict[l])
                else:
                    cats.append("resource")
            return cats


        pm.loc[:, 'src_category'] = get_category(categories1, pm['src_id'].tolist())
        pm.loc[:, 'tgt_category'] = get_category(categories2, pm['tgt_id'].tolist())
        gs.loc[:, 'src_category'] = get_category(categories1, gs['src_id'].tolist())
        gs.loc[:, 'tgt_category'] = get_category(categories2, gs['tgt_id'].tolist())
        pm = pm.loc[pm.src_category == pm.tgt_category]
        gs = gs.loc[gs.src_category == gs.tgt_category]


        CONFIGURATION.log("      --> Applying ontology restrictions: 100% [inactive]")


        CONFIGURATION.log("      --> Calculating implicit features: 0% [inactive]", end="\r")


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
            #src_origin = np.full((len(df), src_dim), 0.0000001)
            #tgt_origin = np.full((len(df), tgt_dim), 0.0000001)
            #df['src_angle_to_origin'] = paired_cosine_distances(tgt_origin,a)
            #df['tgt_angle_to_origin'] = paired_cosine_distances(src_origin,b)
            #df['src_veclen'] = length(a)
            #df['tgt_veclen'] = length(b)
            df['src_tgt_veclen'] = paired_euclidean_distances(a,b)#.diagonal()#length(a-b)
            df.head()

            df.fillna(0, inplace = True)
            return df


        # In[322]:

        pm = extend_features(pm)
        gs = extend_features(gs)
        #oaei_gold_standard3 = extend_features(oaei_gold_standard3)


        # In[310]:



        def jacc(s,t, n=3):
            s = labels1[s]
            t = labels2[t]
            t = set([t[i:i+n] for i in range(len(t)-n+1)])
            s = set([s[i:i+n] for i in range(len(s)-n+1)])
            return 1-len([gram for gram in s if gram in t])/max(len(s), len(t))


        memo = {}
        def lev(s,t, n=3):
            memo = {}
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
        #pm['syntactic_diff'] = pm.apply(lambda row: jacc(row['src_id'], row['tgt_id']), axis=1)
        pm['plus_diff'] = pm.apply(lambda row: jacc(row['src_id'], row['tgt_id']), axis=1)
        gs['plus_diff'] = gs.apply(lambda row: jacc(row['src_id'], row['tgt_id']), axis=1)


        CONFIGURATION.log("      --> Calculating implicit features: 100% [inactive]")



        CONFIGURATION.log("      --> Performing machine learning step: 0% [inactive]", end="\r")
        cols = [col for col in gs.columns if col not in ['target','src_id','tgt_id','src_category','tgt_category', 'label_x','label_y']]#['src_tgt_angle', 'src_tgt_veclen', 'plus_diff', 'syntactic_diff']
        X, y = gs[cols], gs.target
        weight_ratio = float(len(y[y == 0]))/float(len(y[y == 1]))
        w_array = np.array([1]*y.shape[0])
        w_array[y==1] = weight_ratio*2.0
        w_array[y==0] = ( 1 - weight_ratio )
        clf = XGBClassifier().fit(X, y, sample_weight=w_array)
        #random_state=0, solver='lbfgs', multi_class='ovr', class_weight={1:0.1,0:0.9}).fit(X, y)
        pm = pm.loc[clf.predict(X)==1]
        CONFIGURATION.log("      --> Performing machine learning step: 100% [inactive]")


        pm.loc[:, 'total_score'] = 0
        pm.loc[:, 'syntax_score'] = 0
        pm.loc[:, 'confidence_score'] = 0
        pm.loc[:, 'euclid_score'] = 0
        pm.loc[:, 'probability_score'] = 0
        pm.loc[:, 'cos_score'] = 0

        syntax_score = np.zeros(len(pm))
        confidence_score = np.zeros(len(pm))
        euclid_score = np.zeros(len(pm))
        probability_score = np.zeros(len(pm))
        cos_score = np.zeros(len(pm))


        pm = pm.reset_index()
        pm = pm.sort_index()
        ind = pm.index.values
        pm2 = pm.set_index('src_id')
        pm2.loc[:, 'former_index'] = ind
        pm2 = pm2[['former_index']]
        pm2 = pm2.sort_index()


        syntax_score = np.zeros(len(pm))
        confidence_score = np.zeros(len(pm))
        euclid_score = np.zeros(len(pm))
        probability_score = np.zeros(len(pm))
        cos_score = np.zeros(len(pm))

        progress = 0
        total = len(set(pm.src_id))
        total = total + len(set(pm.tgt_id))
        for nodeid in set(pm.src_id):
                        inds = pm2.loc[nodeid,"former_index"].tolist()
                        if type(inds) == list:
                            x = pm.loc[inds]
                        else:
                            x = pm.loc[inds].to_frame().transpose()

                        progress += 1
                        CONFIGURATION.log("      --> Calculating final scores: " + str(int(100*progress/total)) + "% [active]", end="\r")


                        ctr = 1
                        x = x.sort_values(by=['src_tgt_angle'], ascending=True)
                        for index, row in x.iterrows():
                            cos_score[index] = row['cos_score'] + 1/ctr
                            ctr += 1

                        x = x.sort_values(by=['src_tgt_veclen'], ascending=True)
                        ctr = 1
                        ##x.columns = ['euclid_sim' if col==0 else col for col in x.columns]
                        for index, row in x.iterrows():
                            euclid_score[index] = row['euclid_score'] + 1/ctr
                            ctr += 1

                        x = x.sort_values(by=['plus_diff'], ascending=True)
                        ctr = 1
                        ##x.columns = ['euclid_sim' if col==0 else col for col in x.columns]
                        for index, row in x.iterrows():
                            syntax_score[index] = row['syntax_score'] + 1/ctr
                            ctr += 1

        pm.loc[:, 'syntax_score'] = syntax_score
        pm.loc[:, 'confidence_score'] = confidence_score
        pm.loc[:, 'euclid_score'] = euclid_score
        pm.loc[:, 'probability_score'] = probability_score
        pm.loc[:, 'cos_score'] = cos_score

        pm = pm.reset_index()
        pm = pm.sort_index()
        ind = pm.index.values
        pm3 = pm.set_index('tgt_id')
        pm3.loc[:, 'former_index'] = ind
        pm3 = pm3[['former_index']]
        pm3 = pm3.sort_index()

        syntax_score = np.zeros(len(pm))
        confidence_score = np.zeros(len(pm))
        euclid_score = np.zeros(len(pm))
        probability_score = np.zeros(len(pm))
        cos_score = np.zeros(len(pm))

        for nodeid in set(pm.tgt_id):
                        inds = pm3.loc[nodeid,"former_index"].tolist()
                        if type(inds) == list:
                            x = pm.loc[inds]
                        else:
                            x = pm.loc[inds].to_frame().transpose()

                        progress += 1
                        CONFIGURATION.log("      --> Calculating final scores: " + str(int(100*progress/total)) + "% [active]", end="\r")


                        ctr = 1
                        x = x.sort_values(by=['src_tgt_angle'], ascending=True)
                        for index, row in x.iterrows():
                            cos_score[index] = row['cos_score'] + 1/ctr
                            ctr += 1

                        x = x.sort_values(by=['src_tgt_veclen'], ascending=True)
                        ctr = 1
                        ##x.columns = ['euclid_sim' if col==0 else col for col in x.columns]
                        for index, row in x.iterrows():
                            euclid_score[index] = row['euclid_score'] + 1/ctr
                            ctr += 1

                        x = x.sort_values(by=['plus_diff'], ascending=True)
                        ctr = 1
                        ##x.columns = ['euclid_sim' if col==0 else col for col in x.columns]
                        for index, row in x.iterrows():
                            syntax_score[index] = row['syntax_score'] + 1/ctr
                            ctr += 1


        pm.loc[:, 'syntax_score'] = syntax_score
        pm.loc[:, 'confidence_score'] = confidence_score
        pm.loc[:, 'euclid_score'] = euclid_score
        pm.loc[:, 'probability_score'] = probability_score
        pm.loc[:, 'cos_score'] = cos_score
        pm.loc[:, 'total_score'] = pm['syntax_score'] + pm['euclid_score'] + pm['probability_score'] + pm['confidence_score'] + \
                                  pm['cos_score']

        CONFIGURATION.log("      --> Calculating final scores: 100% [active]")
        return pm

def match(pm, graph1, graph2):
    # In[263]:

    #matchinpm = matchinpm_saved
    #matchinpm = pm

#
    ## In[264]:
#
    #def listify(l):
    #    if type(l) == list:
    #        return l
    #    else:
    #        return [l]
#
    #pm.sort_values(by=['total_score', 'src_tgt_angle'], ascending=[False, True], inplace=True
    #married_matchinpm = list()
    #CONFIGURATION.log("sort done")
    #CONFIGURATION.log(len(pm)))
    #pm.loc[:, 'married'] = False
    #smallerpm = pm["married"]
    #while len(smallerpm.loc[smallerpm == False]) > 0:
    #    row = pm.loc[smallerpm.loc[(smallerpm == False)].head(1).index.values[0], ['src_id', 'tgt_id']]
    #    married_matchinpm.append([row['src_id'], row['tgt_id']])
    #    l1 = listify(pm2.loc[(row.src_id), "former_index"].tolist())
    #    l2 = listify(pm3.loc[(row.tgt_id), "former_index"].tolist())
    #    l3 = l1 + l2
    #    # negated_select_on_pm = np.zeros(len(pm))
    #    # for i in l3:
    #    #    if i>len(pm):
    #    #        break
    #    #    negated_select_on_pm[i] = 1
    #    # negated_select_on_pm = negated_select_on_pm==0
    #    smallerpm.loc[l3] = True
    #    CONFIGURATION.log(str(len(smallerpm.loc[smallerpm == False])) + " left     ", end="\r")
#
    #married_matchinpm = pd.DataFrame(married_matchinpm)
    #married_matchinpm.columns = ['src_id', 'tgt_id']
    #married_matchinpm.head()
#
    #married_matchinpm[['src_id','tgt_id']].to_csv(basedir+"married_matchinpm.csv", encoding="UTF-8", sep="\t")
    #PredictionToXMLConverter.interface(PipelineDataTuple(graph1, graph2), PipelineDataTuple('married_matchinpm.csv'), CONFIGURATION)


    CONFIGURATION.log("      --> Peforming stable marriage: 0% [active]", end="\r")

    d2 = dict()
    d3 = dict()
    for index, row in pm.iterrows():
        if row['src_id'] in d2.keys():
            d2[row['src_id']].append(index)
        else:
            d2[row['src_id']] = [index]
        if row['tgt_id'] in d3.keys():
            d3[row['tgt_id']].append(index)
        else:
            d3[row['tgt_id']] = [index]

    pm.sort_values(by=['total_score', 'src_tgt_angle'], ascending=[False, True], inplace=True)
    values = dict()
    inverted_values = dict()
    tmp = pm[['src_id', 'tgt_id']].values.tolist()
    left_for_mapping = pm.index.values.tolist()
    for i in range(len(pm.index.values.tolist())):
        values[left_for_mapping[i]] = tmp[i]
        if tmp[i][0] in inverted_values.keys():
            inverted_values[tmp[i][0]].append(left_for_mapping[i])
        else:
            inverted_values[tmp[i][0]] = [left_for_mapping[i]]
        if tmp[i][1] in inverted_values.keys():
            inverted_values[tmp[i][1]].append(left_for_mapping[i])
        else:
            inverted_values[tmp[i][1]] = [left_for_mapping[i]]

    mem = len(left_for_mapping)
    total = len(left_for_mapping)
    married_matchinpm = list()
    while len(left_for_mapping) > 0:
        ind = left_for_mapping[0]
        a = values[ind][0]
        b = values[ind][1]
        married_matchinpm.append([a, b])
        for x in inverted_values[a]:
            if x in left_for_mapping:
                left_for_mapping.remove(x)
        for x in inverted_values[b]:
            if x in left_for_mapping:
                left_for_mapping.remove(x)
        if mem - len(left_for_mapping) > 100:
            CONFIGURATION.log("      --> Peforming stable marriage: "+str(int(100*(total-len(left_for_mapping))/total))+"% [active]", end="\r")
            mem = len(left_for_mapping)

    CONFIGURATION.log("      --> Peforming stable marriage: 100% [active]")
    married_matchinpm = pd.DataFrame(married_matchinpm)
    married_matchinpm.columns = ['src_id', 'tgt_id']
    married_matchinpm.head()

    CONFIGURATION.log("      --> Storing results: 0% [inactive]", end="\r")
    married_matchinpm[['src_id','tgt_id']].to_csv(CONFIGURATION.rundir+"married_matchinpm.csv", encoding="UTF-8", sep="\t")
    PredictionToXMLConverter.interface(PipelineDataTuple(graph1, graph2), PipelineDataTuple('married_matchinpm.csv'), CONFIGURATION)

    CONFIGURATION.log("      --> Storing results: 100% [inactive]")


def interface(main_input, arpm, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    graph1 = main_input.get(0)
    graph2 = main_input.get(1)
    assert graph1 is not None, "Graph (1) not found in " + os.path.basename(sys.argv[0])
    assert graph2 is not None, "Graph (2) not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.logfile is not None, "Path to logfile not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.name is not None, "Test config name not found in " + os.path.basename(sys.argv[0])
    return exec(graph1, graph2)
