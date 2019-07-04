import pandas as pd
import numpy as np
import ntpath


from sklearn.metrics import classification_report
from cle.configurations.PipelineTools import PipelineDataTuple
from cle.matcher import Matchdata_Saver, PredictionToXMLConverter
import sys
import os
from joblib import dump, load
from xgboost import XGBClassifier

from gensim.models import Doc2Vec, Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import editdistance
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from sklearn.linear_model import LogisticRegression


global CONFIGURATION

def exec(graph1, graph2):


        # In[270]:
        additional_features = None
        progress = 0

        def mergedf(df1, df2):
            if df1 is None:
                return df2
            else:
                return df1.append(df2, ignore_index=True)

        basedir = CONFIGURATION.rundir
        current_process_dir = basedir
        dirpath = basedir
        all_possible_matches_path = CONFIGURATION.gold_mapping.raw_testsets[0]

        documents_ids_A = dict()
        documents_ids_B = dict()
        all_possible_matches = dict()
        all_nodeids = set()
        with open(all_possible_matches_path, encoding="UTF-8") as f:
            for line in f:
                line = line.replace("\n","").split("\t")
                all_nodeids.add(line[0])
                if line[0] in all_possible_matches.keys():
                    all_possible_matches[line[0]].add(line[1])
                else:
                    all_possible_matches[line[0]] = set([line[1]])

                if line[1] in all_possible_matches.keys():
                    all_possible_matches[line[1]].add(line[0])
                else:
                    all_possible_matches[line[1]] = set([line[0]])

        possible_matches = CONFIGURATION.gold_mapping.prepared_testsets[0]#pd.read_csv(dirpath + "possible_matches.csv-strcombined.csv", sep=",", encoding="UTF-8")
        #possible_matches_ids = pd.read_csv(dirpath + "possible_matches.csv-strcombined_ids.csv", sep=",", encoding="UTF-8")
        #possible_matches = possible_matches.merge(possible_matches_ids, left_on=['Unnamed: 0'], right_on=['Unnamed: 0'])


        oaei_gold_standard3 = CONFIGURATION.gold_mapping.prepared_trainsets[0]#pd.read_csv(dirpath + "oaei_gold_standard3.csv-strcombined.csv", sep=",", encoding="UTF-8")
        #oaei_gold_standard3_ids = pd.read_csv(dirpath + "oaei_gold_standard3.csv-strcombined_ids.csv", sep=",", encoding="UTF-8")
        #oaei_gold_standard3 = oaei_gold_standard3.merge(oaei_gold_standard3_ids, left_on=['Unnamed: 0'], right_on=['Unnamed: 0'])




        def get_possible_matches(nid):
                final_matches = list(all_possible_matches[nid])
                #if nid in resources.label.tolist():
                #    for m in matches:
                #        if m in resources.label.tolist():
                #            pass
                #            final_matches.append(m)
#
                #if nid in classes.label.tolist():
                #    for m in matches:
                #        if m in classes.label.tolist():
                #            final_matches.append(m)
#
                #if nid in properties.label.tolist():
                #    for m in matches:
                #        if m in properties.label.tolist():
                #            final_matches.append(m)

                return final_matches


        # In[311]:


        def get_training_material(nid):
            res = list()
            with open(dirpath+"w2v_training_material.csv", mode="r", encoding="UTF-8") as f:
                for line in f:
                    if nodeid in line.split(" "):
                        res = res + line.split(" ")
                return list(set(res))
        model = Word2Vec.load(dirpath+"w2v.model")

        total = len(all_nodeids)
        matchings = None
        with open(dirpath+'additional_features.csv', mode="w+", encoding="UTF-8") as f:
            for nodeid in all_nodeids:

                possible_matches_for_nodeid = possible_matches.loc[((possible_matches.src_id==nodeid) & (possible_matches.tgt_id.isin(get_possible_matches(nodeid))))]



                progress += 1
                if len(get_possible_matches(nodeid))<1:
                    continue


                #vecs = model.wv[get_possible_matches(nodeid)]
                def edits(v1, v2s):
                    res = list()
                    v1 = v1.split("/")[-1]
                    for v2 in v2s:
                        v2 = v2.split("/")[-1]
                        res.append(editdistance.eval(v1, v2)/min(len(v1), len(v2)))
                    return np.array([res])
                #x = edits(nodeid, get_possible_matches(nodeid))
                #x = np.concatenate((x, np.array([get_possible_matches(nodeid)])), axis=0)
                #sorted_x = pd.DataFrame(x).T.sort_values(by=[0], ascending=True)
                sorted_x = possible_matches_for_nodeid.sort_values(by=['syntactic_diff'], ascending=True)
                sorted_x.loc[:,'syntax_score'] = 0
                ctr = 1
                #sorted_x.columns = ['syntax_diff' if col==0 else col for col in sorted_x.columns]
                for index, row in sorted_x.iterrows():
                    #print(row[1] + " - " + str(row['syntax_diff']))
                    sorted_x.loc[index, 'syntax_score'] = row['syntax_score'] + 1/ctr
                    ctr += 1




                #print('Closest in sum:')
                x = sorted_x
                x.loc[:,'total_score'] = x['cos_score'] + x['syntax_score'] + x['euclid_score'] + x['probability_score']
                sorted_x = x.sort_values(by=['total_score'], ascending=False)
                #sorted_x.columns = ['tgt_id' if col==1 else col for col in sorted_x.columns]
                for index, row in sorted_x.iterrows():#sorted_x.loc[sorted_x.total_score == max(sorted_x.total_score.values),:].iterrows():
                    matching_pair = pd.DataFrame([sorted_x.loc[index]])
                    matching_pair.loc[:,'src_id'] = nodeid
                    #print(nodeid + "\t" + row[1] + "\t" + str(row['total_score']) + "\t" + str(row['cos_score']) + "\t" + str(row['euclid_score']))
                    matchings = mergedf(matchings, matching_pair)



                print("         Computing syntax-ranks: " + str(int(100*progress/total)) + "%.", end='\r')

        print("         Computing syntax-ranks: 100%")



        matchings.to_csv(dirpath+"additional_features.csv")

        cols = [col for col in oaei_gold_standard3.columns if col not in ['label','src_id','tgt_id','src_category','tgt_category']]#['src_tgt_angle', 'src_tgt_veclen', 'plus_diff', 'syntactic_diff']
        X, y = oaei_gold_standard3[cols], oaei_gold_standard3.label
        clf = XGBClassifier().fit(X, y)
        #random_state=0, solver='lbfgs', multi_class='ovr', class_weight={1:0.1,0:0.9}).fit(X, y)

        X, y = matchings[cols], matchings.label
        matchings = matchings.loc[clf.predict(X)==1]


        try:
            CONFIGURATION.log("\nStableRankMatcher - logistic regression hyperparameters:\n")
            CONFIGURATION.log("Coefficients: " + str(clf.coef_) + " for " + str(list(set(cols))) +"\n")
            CONFIGURATION.log("Intercept: " + str(clf.intercept_) + "\n")
        except:
            pass
        matchings.to_csv(dirpath+"remaining_matchings.csv", sep="\t")


        matchings = matchings.sort_values(by=['total_score','src_tgt_angle'], ascending=[False, False])
        married_matchings = None
        ctr = 0
        while len(matchings) > 0:
                ctr += 1
                row = matchings.head(1)
                married_matchings = mergedf(married_matchings, pd.DataFrame(row))
                matchings = matchings.loc[~(matchings.src_id == row.src_id.values[0]) & ~(matchings.tgt_id == row.tgt_id.values[0])]

        if married_matchings is not None:
            married_matchings[['src_id','tgt_id']].to_csv(dirpath+"married_matchings.csv", sep="\t", index=False)


            PredictionToXMLConverter.interface(PipelineDataTuple(graph1, graph2), PipelineDataTuple('married_matchings.csv'), CONFIGURATION)




            CONFIGURATION.log("\n\nStableRankMatcher - logistic regression performance:\n")
            CONFIGURATION.log(classification_report(np.array(y), clf.predict(X)))

            if len(married_matchings)>0:
                married_matchings.loc[:,'married'] = 'x'
                possible_matches = possible_matches.merge(married_matchings[['src_id','tgt_id', 'married','total_score']], left_on=['src_id', 'tgt_id'], right_on=['src_id', 'tgt_id'], how='left')
                possible_matches.loc[:, 'prediction'] = 0
                possible_matches.loc[~(possible_matches.married.isna()), 'prediction'] = 1
                CONFIGURATION.log("\n\nStableRankMatcher - marriage performance:\n")
                CONFIGURATION.log(classification_report(np.array(possible_matches.label), np.array(possible_matches.prediction)))
            else:
                CONFIGURATION.log("\n\nStableRankEmbeddingsMatcher - marriage performance: 00.00 (no matches found)\n")



        return PipelineDataTuple(graph1, graph2)


def interface(main_input, args, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    graph1 = main_input.get(0)
    graph2 = main_input.get(1)
    assert graph1 is not None, "Graph (1) not found in " + os.path.basename(sys.argv[0])
    assert graph2 is not None, "Graph (2) not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.gold_mapping is not None, "Path to gold standard file not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.logfile is not None, "Path to logfile not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.name is not None, "Test config name not found in " + os.path.basename(sys.argv[0])
    return exec(graph1, graph2)
