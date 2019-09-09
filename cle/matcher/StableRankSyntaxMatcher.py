import pandas as pd
import numpy as np
import ntpath

from sklearn.metrics import classification_report

from configurations.PipelineTools import PipelineDataTuple
from matcher import Matchdata_Saver
import sys
import os
from joblib import dump, load
from xgboost import XGBClassifier
from matcher import PredictionToXMLConverter

from gensim.models import Doc2Vec, Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import editdistance
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from sklearn.linear_model import LogisticRegression


global CONFIGURATION

def exec(graph1, graph2):

        def mergedf(df1, df2):
            if df1 is None:
                return df2
            else:
                return df1.append(df2, ignore_index=True)


        basedir = CONFIGURATION.rundir
        current_process_dir = basedir
        dirpath = basedir

        possible_matches = CONFIGURATION.gold_mapping.prepared_testsets[0]#pd.read_csv(dirpath + "possible_matches.csv-strcombined.csv", sep=",")
        #possible_matches_ids = pd.read_csv(dirpath + "possible_matches.csv-strcombined_ids.csv", sep=",")
        #possible_matches = possible_matches.merge(possible_matches_ids, left_on=['Unnamed: 0'], right_on=['Unnamed: 0'])


        oaei_gold_standard3 = CONFIGURATION.gold_mapping.prepared_trainsets[0]#pd.read_csv(dirpath + "oaei_gold_standard3.csv-strcombined.csv", sep=",")
        #oaei_gold_standard3_ids = pd.read_csv(dirpath + "oaei_gold_standard3.csv-strcombined_ids.csv", sep=",")
        #oaei_gold_standard3 = oaei_gold_standard3.merge(oaei_gold_standard3_ids, left_on=['Unnamed: 0'], right_on=['Unnamed: 0'])

        cols = ['syntactic_diff']
        X, y = oaei_gold_standard3[cols], oaei_gold_standard3.label
        clf = XGBClassifier().fit(X, y)
        #random_state=0, solver='lbfgs', multi_class='ovr', class_weight={1:0.5,0:0.5}).fit(X, y)

        X, y = possible_matches[cols], possible_matches.label
        matchings = possible_matches.loc[clf.predict(X)==1]

        try:
            CONFIGURATION.log("\nStableRankMatcher - logistic regression hyperparameters:\n")
            CONFIGURATION.log("Coefficients: " + str(clf.coef_) + " for " + str(list(set(cols))) +"\n")
            CONFIGURATION.log("Intercept: " + str(clf.intercept_) + "\n")
        except:
            pass
        matchings.to_csv(dirpath+"remaining_matchings.csv", sep="\t")

        matchings = matchings.sort_values(by=['syntactic_diff'], ascending=[True])
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

            CONFIGURATION.log("\n\nStableRankEmbeddingsMatcher - logistic regression performance:\n")
            CONFIGURATION.log(classification_report(np.array(y), clf.predict(X)))

            if len(married_matchings)>0:
                married_matchings.loc[:,'married'] = 'x'
                possible_matches = possible_matches.merge(married_matchings[['src_id','tgt_id', 'married','total_score']], left_on=['src_id', 'tgt_id'], right_on=['src_id', 'tgt_id'], how='left')
                possible_matches.loc[:, 'prediction'] = 0
                possible_matches.loc[~(possible_matches.married.isna()), 'prediction'] = 1
                CONFIGURATION.log("\n\nStableRankEmbeddingsMatcher - marriage performance:\n")
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
