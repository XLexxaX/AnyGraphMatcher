import pandas as pd
import numpy as np
import ntpath

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

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
from sklearn import preprocessing
import numpy as np
min_max_scaler = preprocessing.MinMaxScaler()


global CONFIGURATION

def exec():

    match(prepare())
    return PipelineDataTuple(None)


def prepare():
        #EmbeddingSaver.interface(PipelineDataTuple(graph1, graph2), None, CONFIGURATION)
        #CONFIGURATION.log("      --> Cleaning memory: 0% [inactive]", end="\r")
        #RAMCleaner.interface(PipelineDataTuple(graph1, graph2), None, CONFIGURATION)

        #CONFIGURATION.log("      --> Cleaning memory: 100% [inactive]")

        basedir = CONFIGURATION.rundir



        if os.path.isfile(CONFIGURATION.gold_mapping.raw_testsets[0]) and os.path.getsize(CONFIGURATION.gold_mapping.raw_testsets[0]) > 0:
            pm = pd.read_csv(CONFIGURATION.gold_mapping.raw_testsets[0], sep="\t", header=None, encoding=CONFIGURATION.encoding)
            pm.columns = ['src_id','tgt_id']
        else:
            return None

        embs = pd.read_csv(basedir+"stratified_embeddings.csv", sep="\t", encoding=CONFIGURATION.encoding)
        #embs = embs[[col for col in embs.columns if re.match('x\d+', col) is not None]+['label']]
        #embs.columns = ["src_" + str(col) for col in [re.search("\d+", col).group(0) for col in embs.columns if re.match('src_\d+', col) is not None]] + ['label']
        pm = pm.merge(embs, left_on=['src_id'], right_on=['label'])
        embs.columns = ["tgt_" + str(col) for col in [re.search("\d+", col).group(0) for col in embs.columns if
                                                      re.match('src_\d+', col) is not None]] + ['label']
        pm = pm.merge(embs, left_on=['tgt_id'], right_on=['label'])



        if os.path.isfile(CONFIGURATION.gold_mapping.raw_testsets[0]) and os.path.getsize(CONFIGURATION.gold_mapping.raw_testsets[0]) > 0:
            gs = pd.read_csv(CONFIGURATION.gold_mapping.raw_trainsets[0], sep="\t", header=None, encoding=CONFIGURATION.encoding)
            gs.columns = ['src_id','tgt_id','target']
        else:
            return None
        embs.columns = ["src_" + str(col) for col in [re.search("\d+", col).group(0) for col in embs.columns if re.match('tgt_\d+', col) is not None]] + ['label']

        gs = gs.merge(embs, left_on=['src_id'], right_on=['label'])
        embs.columns = ["tgt_" + str(col) for col in [re.search("\d+", col).group(0) for col in embs.columns if
                                                      re.match('src_\d+', col) is not None]] + ['label']

        gs = gs.merge(embs, left_on=['tgt_id'], right_on=['label'])



        CONFIGURATION.log("      --> Applying ontology restrictions: 0% [inactive]", end="\r")

        labels1 = dict()
        categories1 = dict()
        with open(CONFIGURATION.src_triples, mode="r", encoding=CONFIGURATION.encoding) as f:
            for line in f:
                for label_property in CONFIGURATION.properties.src_label_properties:
                    if " <" + label_property + "> " in line:
                        line = line.replace("<","").replace(">","").replace(" .\n","").split(" "+label_property+" ")
                        if line[0] in line1.keys():
                            labels1[line[0]][label_property] =  line[1]
                        else:
                            dct = dict()
                            dct[label_property] = line[1]
                            labels1[line[0]] = dct
                if " <" + CONFIGURATION.properties.category_property + "> " in line:
                    line = line.replace("<","").replace(">","").replace(" .\n","").split(" "+CONFIGURATION.properties.category_property+" ")
                    if line[1] not in [CONFIGURATION.properties.class_descriptor,CONFIGURATION.properties.property_descriptor]:
                        categories1[line[0]] = 'resource'
                    else:
                         categories1[line[0]] = line[1]
        labels2 = dict()
        categories2 = dict()
        with open(CONFIGURATION.tgt_triples, mode="r", encoding=CONFIGURATION.encoding) as f:
                    for line in f:
                        for label_property in CONFIGURATION.properties.tgt_label_properties:
                            if " <" + label_property + "> " in line:
                                line = line.replace("<","").replace(">","").replace(" .\n","").split(" "+label_property+" ")
                                if line[0] in line1.keys():
                                    labels1[line[0]][label_property] =  line[1]
                                else:
                                    dct = dict()
                                    dct[label_property] = line[1]
                                    labels1[line[0]] = dct
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
            s_prop = ""
            t_prop = ""
            for label_property in set(CONFIGURATION.properties.src_label_properties + CONFIGURATION.properties.tgt_label_properties):
                if label_property in labels1[s].keys() and label_property in labels2[t].keys():
                    s_prop = s_prop + labels1[s][label_property]
                    t_prop = t_prop + labels2[t][label_property]
            t = set([t_prop[i:i+n] for i in range(len(t_prop)-n+1)])
            s = set([s_prop[i:i+n] for i in range(len(s_prop)-n+1)])
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
        pm['plus_diff'] = pm.apply(lambda row: jacc(row['src_id'].str.lower(), row['tgt_id'].str.lower()), axis=1)
        gs['plus_diff'] = gs.apply(lambda row: jacc(row['src_id'].str.lower(), row['tgt_id'].str.lower()), axis=1)


        CONFIGURATION.log("      --> Calculating implicit features: 100% [inactive]")



        CONFIGURATION.log("      --> Performing machine learning step: 0% [inactive]", end="\r")
        cols = [col for col in gs.columns if col not in ['target','src_id','tgt_id','src_category','tgt_category', 'label_x','label_y']]#['src_tgt_angle', 'src_tgt_veclen', 'plus_diff', 'syntactic_diff']
        X, y = gs[cols], gs.target
        weight_ratio = float(len(y[y == 0]))/float(len(y[y == 1]))
        w_array = np.array([1]*y.shape[0])
        w_array[y==1] = weight_ratio*1.0
        w_array[y==0] = ( 1 - weight_ratio )
        clf = XGBClassifier().fit(X, y, sample_weight=w_array)
        #random_state=0, solver='lbfgs', multi_class='ovr', class_weight={1:0.1,0:0.9}).fit(X, y)
        X = pm[cols]
        pm = pm.loc[clf.predict(X)==1]
        pm = pm.loc[X.plus_diff < 0.01]
        CONFIGURATION.log("      --> Performing machine learning step: 100% [inactive]")



        CONFIGURATION.log("      --> Performing 3-pair restriction: 0% [inactive]")
        pm.loc[:, 'delete_flag'] = False
        for nodeid in set(pm.src_id):
                                x = pm.loc[pm.src_id==nodeid]
                                x = x.sort_values(by=['plus_diff'], ascending=True)
                                pm.loc[x[2:].index, 'delete_flag'] = True
        for nodeid in set(pm.tgt_id):
                                x = pm.loc[pm.tgt_id==nodeid]
                                x = x.sort_values(by=['plus_diff'], ascending=True)
                                pm.loc[x[2:].index, 'delete_flag'] = True
        pm = pm.loc[pm.delete_flag==False]
        CONFIGURATION.log("      --> Performing 3-pair restriction: 100% [inactive]")

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

                        x = x.sort_values(by=['plus_diff'], ascending=True)
                        x = x.head(3)

                        progress += 1
                        CONFIGURATION.log("      --> Calculating final scores: " + str(int(100*progress/total)) + "% [active]", end="\r")


                        multiplier = len(x)



                        #x['src_tgt_angle'] = min_max_scaler.fit_transform(tmp[['plus_diff']])
                        #x['src_tgt_veclen'] = min_max_scaler.fit_transform(tmp[['plus_diff']])
                        #x['plus_diff'] = min_max_scaler.fit_transform(tmp[['plus_diff']])

                        ctr = 1
                        x = x.sort_values(by=['src_tgt_angle'], ascending=False)
                        for index, row in x.iterrows():
                            if multiplier == 1:
                                cos_score[index] = row['cos_score'] + 0.5
                            else:
                                cos_score[index] = row['cos_score'] + (multiplier-ctr)*(1/(multiplier-1))#1/ctr
                            ctr += 1

                        x = x.sort_values(by=['src_tgt_veclen'], ascending=True)
                        ctr = 1
                        ##x.columns = ['euclid_sim' if col==0 else col for col in x.columns]
                        for index, row in x.iterrows():
                            if multiplier == 1:
                                euclid_score[index] = row['euclid_score'] + 0.5#1/ctr
                            else:
                                euclid_score[index] = row['euclid_score'] + (multiplier-ctr)*(1/(multiplier-1))#1/ctr
                            ctr += 1

                        x = x.sort_values(by=['plus_diff'], ascending=True)
                        ctr = 1
                        ##x.columns = ['euclid_sim' if col==0 else col for col in x.columns]
                        lastscore = -1
                        lastvalue = -1
                        for index, row in x.iterrows():
                            if row['plus_diff'] == lastvalue:
                                syntax_score[index] = row['syntax_score'] + lastscore
                            elif multiplier == 1:
                                syntax_score[index] = row['syntax_score'] + 0.5#1/ctr
                                lastscore = 0.5
                            else:
                                syntax_score[index] = row['syntax_score'] + (multiplier-ctr)*(1/(multiplier-1))#1/ctr
                                lastscore =  (multiplier-ctr)*(1/(multiplier-1))
                            lastvalue = row['plus_diff']
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


                        x = x.sort_values(by=['plus_diff'], ascending=True)
                        x = x.head(3)


                        progress += 1
                        CONFIGURATION.log("      --> Calculating final scores: " + str(int(100*progress/total)) + "% [active]", end="\r")


                        multiplier = len(x)


                        #x['src_tgt_angle'] = min_max_scaler.fit_transform(tmp[['plus_diff']])
                        #x['src_tgt_veclen'] = min_max_scaler.fit_transform(tmp[['plus_diff']])
                        #x['plus_diff'] = min_max_scaler.fit_transform(tmp[['plus_diff']])

                        ctr = 1
                        x = x.sort_values(by=['src_tgt_angle'], ascending=False)
                        for index, row in x.iterrows():
                            if multiplier == 1:
                                cos_score[index] = row['cos_score'] + 0.5
                            else:
                                cos_score[index] = row['cos_score'] + (multiplier-ctr)*(1/(multiplier-1))#1/ctr
                            ctr += 1

                        x = x.sort_values(by=['src_tgt_veclen'], ascending=True)
                        ctr = 1
                        ##x.columns = ['euclid_sim' if col==0 else col for col in x.columns]
                        for index, row in x.iterrows():
                            if multiplier == 1:
                                euclid_score[index] = row['euclid_score'] + 0.5#1/ctr
                            else:
                                euclid_score[index] = row['euclid_score'] + (multiplier-ctr)*(1/(multiplier-1))#1/ctr
                            ctr += 1

                        x = x.sort_values(by=['plus_diff'], ascending=True)
                        ctr = 1
                        ##x.columns = ['euclid_sim' if col==0 else col for col in x.columns]
                        lastscore = -1
                        lastvalue = -1
                        for index, row in x.iterrows():
                            if row['plus_diff'] == lastvalue:
                                syntax_score[index] = row['syntax_score'] + lastscore
                            elif multiplier == 1:
                                syntax_score[index] = row['syntax_score'] + 0.5#1/ctr
                                lastscore = 0.5
                            else:
                                syntax_score[index] = row['syntax_score'] + (multiplier-ctr)*(1/(multiplier-1))#1/ctr
                                lastscore =  (multiplier-ctr)*(1/(multiplier-1))
                            lastvalue = row['plus_diff']
                            ctr += 1


        pm.loc[:, 'syntax_score'] = syntax_score
        pm.loc[:, 'confidence_score'] = confidence_score
        pm.loc[:, 'euclid_score'] = euclid_score
        pm.loc[:, 'probability_score'] = probability_score
        pm.loc[:, 'cos_score'] = cos_score
        pm.loc[:, 'total_score'] = 4*pm['syntax_score'] + pm['euclid_score'] + 2*pm['cos_score'] #pm['probability_score'] + pm['confidence_score'] + \
                                  #pm['cos_score']

        pm.to_csv(CONFIGURATION.rundir + "fullfeatures.csv", encoding="UTF-8", sep="\t")

        CONFIGURATION.log("      --> Calculating final scores: 100% [active]")
        return pm

def match(pm):
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
    #married_matchinpm[['src_id','tgt_id']].to_csv(basedir+"married_matchinpm.csv", sep="\t")
    #PredictionToXMLConverter.interface(PipelineDataTuple(graph1, graph2), PipelineDataTuple('married_matchinpm.csv'), CONFIGURATION)


    if pm is None:
        CONFIGURATION.log("      --> Got nothing to match. Prefiltering-criteria might be too sharp (e.g. for the OAEI-conference-track).")
        try:
            os.remove(CONFIGURATION.rundir+"married_matchings.csv")
        except:
            pass
        with open(CONFIGURATION.rundir+"married_matchings.csv", mode="w+", encoding=CONFIGURATION.encoding) as f:
            f.write("\tsrc_id\ttgt_id")
        PredictionToXMLConverter.interface(PipelineDataTuple(None, None), PipelineDataTuple('married_matchings.csv'), CONFIGURATION)
        return


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

    pm.sort_values(by=['plus_diff', 'src_tgt_angle', 'src_tgt_veclen'], ascending=[True, False, True], inplace=True)
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
    married_matchings = list()
    while len(left_for_mapping) > 0:
        ind = left_for_mapping[0]
        a = values[ind][0]
        b = values[ind][1]
        married_matchings.append([a, b])
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
    married_matchings = pd.DataFrame(married_matchings)
    married_matchings.columns = ['src_id', 'tgt_id']
    married_matchings.head()

    CONFIGURATION.log("      --> Storing results: 0% [inactive]", end="\r")
    married_matchings[['src_id','tgt_id']].to_csv(CONFIGURATION.rundir+"married_matchings.csv", sep="\t")
    PredictionToXMLConverter.interface(PipelineDataTuple(None, None), PipelineDataTuple('married_matchings.csv'), CONFIGURATION)

    CONFIGURATION.log("      --> Storing results: 100% [inactive]")


def interface(main_input, arpm, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    assert CONFIGURATION.logfile is not None, "Path to logfile not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.name is not None, "Test config name not found in " + os.path.basename(sys.argv[0])
    return exec()
