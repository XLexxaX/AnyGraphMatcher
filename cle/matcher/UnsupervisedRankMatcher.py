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
from joblib import dump, load
from sklearn.metrics.pairwise import *
import re
from gensim.models import Doc2Vec, Word2Vec


global CONFIGURATION

def exec(graph1, graph2):



        EmbeddingSaver.interface(PipelineDataTuple(graph1, graph2), None, CONFIGURATION)
        CONFIGURATION.log("      --> Cleaning memory: 0% [inactive]", end="\r")


        #RAMCleaner.interface(PipelineDataTuple(graph1, graph2), None, CONFIGURATION)
        graph1.reset()
        graph2.reset()

        CONFIGURATION.log("      --> Cleaning memory: 100% [inactive]")

        match(prepare(), graph1, graph2)
        return None


def prepare():




        basedir = CONFIGURATION.rundir

        gs = pd.read_csv(CONFIGURATION.gold_mapping.raw_testsets[0], sep="\t", header=None)
        gs.columns = ['src_id','tgt_id']
        embs = pd.read_csv(basedir+"stratified_embeddings.csv", sep="\t")
        #embs = embs[[col for col in embs.columns if re.match('x\d+', col) is not None]+['label']]
        #embs.columns = ["src_" + str(col) for col in [re.search("\d+", col).group(0) for col in embs.columns if re.match('x\d+', col) is not None]] + ['label']
        gs = gs.merge(embs, left_on=['src_id'], right_on=['label'])
        embs.columns = ["tgt_" + str(col) for col in [re.search("\d+", col).group(0) for col in embs.columns if
                                                      re.match('src_\d+', col) is not None]] + ['label']
        gs = gs.merge(embs, left_on=['tgt_id'], right_on=['label'])


        CONFIGURATION.log("      --> Applying ontology restrictions: 0% [inactive]", end="\r")

        labels1 = dict()
        categories1 = dict()
        with open(CONFIGURATION.src_triples, mode="r") as f:
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
        with open(CONFIGURATION.tgt_triples, mode="r") as f:
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


        gs.loc[:, 'src_category'] = get_category(categories1, gs['src_id'].tolist())
        gs.loc[:, 'tgt_category'] = get_category(categories2, gs['tgt_id'].tolist())
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
        #gs['syntactic_diff'] = gs.apply(lambda row: jacc(row['src_id'], row['tgt_id']), axis=1)
        gs['plus_diff'] = gs.apply(lambda row: jacc(row['src_id'], row['tgt_id']), axis=1)


        CONFIGURATION.log("      --> Calculating implicit features: 100% [inactive]")


        gs.loc[:, 'total_score'] = 0
        gs.loc[:, 'syntax_score'] = 0
        gs.loc[:, 'confidence_score'] = 0
        gs.loc[:, 'euclid_score'] = 0
        gs.loc[:, 'probability_score'] = 0
        gs.loc[:, 'cos_score'] = 0

        syntax_score = np.zeros(len(gs))
        confidence_score = np.zeros(len(gs))
        euclid_score = np.zeros(len(gs))
        probability_score = np.zeros(len(gs))
        cos_score = np.zeros(len(gs))


        gs = gs.reset_index()
        gs = gs.sort_index()
        ind = gs.index.values
        gs2 = gs.set_index('src_id')
        gs2.loc[:, 'former_index'] = ind
        gs2 = gs2[['former_index']]
        gs2 = gs2.sort_index()


        syntax_score = np.zeros(len(gs))
        confidence_score = np.zeros(len(gs))
        euclid_score = np.zeros(len(gs))
        probability_score = np.zeros(len(gs))
        cos_score = np.zeros(len(gs))

        progress = 0
        total = len(set(gs.src_id))
        total = total + len(set(gs.tgt_id))
        for nodeid in set(gs.src_id):
                        inds = gs2.loc[nodeid,"former_index"].tolist()
                        if type(inds) == list:
                            x = gs.loc[inds]
                        else:
                            x = gs.loc[inds].to_frame().transpose()

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

        gs.loc[:, 'syntax_score'] = syntax_score
        gs.loc[:, 'confidence_score'] = confidence_score
        gs.loc[:, 'euclid_score'] = euclid_score
        gs.loc[:, 'probability_score'] = probability_score
        gs.loc[:, 'cos_score'] = cos_score

        gs = gs.reset_index()
        gs = gs.sort_index()
        ind = gs.index.values
        gs3 = gs.set_index('tgt_id')
        gs3.loc[:, 'former_index'] = ind
        gs3 = gs3[['former_index']]
        gs3 = gs3.sort_index()

        syntax_score = np.zeros(len(gs))
        confidence_score = np.zeros(len(gs))
        euclid_score = np.zeros(len(gs))
        probability_score = np.zeros(len(gs))
        cos_score = np.zeros(len(gs))

        for nodeid in set(gs.tgt_id):
                        inds = gs3.loc[nodeid,"former_index"].tolist()
                        if type(inds) == list:
                            x = gs.loc[inds]
                        else:
                            x = gs.loc[inds].to_frame().transpose()

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


        gs.loc[:, 'syntax_score'] = syntax_score
        gs.loc[:, 'confidence_score'] = confidence_score
        gs.loc[:, 'euclid_score'] = euclid_score
        gs.loc[:, 'probability_score'] = probability_score
        gs.loc[:, 'cos_score'] = cos_score
        gs.loc[:, 'total_score'] = gs['syntax_score'] + gs['euclid_score'] + gs['probability_score'] + gs['confidence_score'] + \
                                  gs['cos_score']

        CONFIGURATION.log("      --> Calculating final scores: 100% [active]")
        return gs

def match(gs, graph1, graph2):
    # In[263]:

    #matchings = matchings_saved
    #matchings = gs

#
    ## In[264]:
#
    #def listify(l):
    #    if type(l) == list:
    #        return l
    #    else:
    #        return [l]
#
    #gs.sort_values(by=['total_score', 'src_tgt_angle'], ascending=[False, True], inplace=True
    #married_matchings = list()
    #CONFIGURATION.log("sort done")
    #CONFIGURATION.log(len(gs)))
    #gs.loc[:, 'married'] = False
    #smallergs = gs["married"]
    #while len(smallergs.loc[smallergs == False]) > 0:
    #    row = gs.loc[smallergs.loc[(smallergs == False)].head(1).index.values[0], ['src_id', 'tgt_id']]
    #    married_matchings.append([row['src_id'], row['tgt_id']])
    #    l1 = listify(gs2.loc[(row.src_id), "former_index"].tolist())
    #    l2 = listify(gs3.loc[(row.tgt_id), "former_index"].tolist())
    #    l3 = l1 + l2
    #    # negated_select_on_gs = np.zeros(len(gs))
    #    # for i in l3:
    #    #    if i>len(gs):
    #    #        break
    #    #    negated_select_on_gs[i] = 1
    #    # negated_select_on_gs = negated_select_on_gs==0
    #    smallergs.loc[l3] = True
    #    CONFIGURATION.log(str(len(smallergs.loc[smallergs == False])) + " left     ", end="\r")
#
    #married_matchings = pd.DataFrame(married_matchings)
    #married_matchings.columns = ['src_id', 'tgt_id']
    #married_matchings.head()
#
    #married_matchings[['src_id','tgt_id']].to_csv(basedir+"married_matchings.csv", sep="\t")
    #PredictionToXMLConverter.interface(PipelineDataTuple(graph1, graph2), PipelineDataTuple('married_matchings.csv'), CONFIGURATION)


    CONFIGURATION.log("      --> Peforming stable marriage: 0% [active]", end="\r")

    d2 = dict()
    d3 = dict()
    for index, row in gs.iterrows():
        if row['src_id'] in d2.keys():
            d2[row['src_id']].append(index)
        else:
            d2[row['src_id']] = [index]
        if row['tgt_id'] in d3.keys():
            d3[row['tgt_id']].append(index)
        else:
            d3[row['tgt_id']] = [index]

    gs.sort_values(by=['total_score', 'src_tgt_angle'], ascending=[False, True], inplace=True)
    values = dict()
    inverted_values = dict()
    tmp = gs[['src_id', 'tgt_id']].values.tolist()
    left_for_mapping = gs.index.values.tolist()
    for i in range(len(gs.index.values.tolist())):
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
    PredictionToXMLConverter.interface(PipelineDataTuple(graph1, graph2), PipelineDataTuple('married_matchings.csv'), CONFIGURATION)

    CONFIGURATION.log("      --> Storing results: 100% [inactive]")


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
