import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from pandas_ml import ConfusionMatrix
from matcher.DatasetHelperTools import stream_prepare_data_from_graph

from configurations.PipelineTools import PipelineDataTuple
from matcher import OAEIMatchdata_Saver
import sys
import os
from joblib import dump, load


global CONFIGURATION

def exec(graph1, graph2, ml_model):

    OAEIMatchdata_Saver.interface(PipelineDataTuple(graph1, graph2), None, CONFIGURATION)

    train = None
    for trainset in CONFIGURATION.gold_mapping.prepared_trainsets:
        if train is None:
            train = pd.read_csv(trainset, index_col=['Unnamed: 0'])
        else:
            tmp_train = pd.read_csv(trainset, index_col=['Unnamed: 0'])
            train = train.append(tmp_train, ignore_index=True)

     # stream test data


    # #### Alternative 1: Sample the training data manually.
    #a = train_simple.loc[train_simple['label']==1].sample(n=100, replace=False)
    #b = train_simple.loc[train_simple['label']==0].sample(n=100, replace=False)
    #c = train_hard.loc[train_hard['label']==1].sample(n=0, replace=False)
    #d = train_hard.loc[train_hard['label']==0].sample(n=600, replace=False)
    #train = d.append(c.append(a.append(b, ignore_index=True), ignore_index=True), ignore_index=True)
    cachefile_path = None
    import hashlib
    import re
    cachefile = hashlib.sha256(bytes(re.escape(CONFIGURATION.gold_mapping.raw_testsets[0]), encoding='UTF-8')).hexdigest() + '.cache'
    if os.path.exists(CONFIGURATION.cachedir + cachefile) and CONFIGURATION.use_cache:
        cachefile_path = CONFIGURATION.cachedir + cachefile

    test = stream_prepare_data_from_graph(graph1, graph2, CONFIGURATION.gold_mapping.raw_testsets[0], cachefile_path)


    # ## Prepare train/test/prediction data
    x_train = train.loc[:, train.columns != 'label']
    y_train = train['label']


    # ## Prediction
    model = ml_model#RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0) #LogisticRegression(solver='lbfgs')
    model = model.fit(x_train, y_train)
    #syntactic_model = LogisticRegression(solver='lbfgs')#RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    #syntactic_model = syntactic_model.fit(pd.DataFrame(x_train['syntactic_diff']), y_train)
    dump(model, CONFIGURATION.rundir + 'model.joblib')
    #dump(model, CONFIGURATION.rundir + 'syntactic_model.joblib')

    prediction = list()
    gold = list()
    plus_prediction = list()
    plus_gold = list()
    ctr = 0
    df = None
    for sample in test:
        ctr = ctr + 1
        CONFIGURATION.log(str(ctr))
        if df is None:
            df = sample
        else:
            df = pd.concat((df, sample), axis=1)
    #    ctr = ctr + 1
    #    prediction = prediction + model.predict(sample.loc[:, sample.columns != 'label']).tolist()
    #    gold = gold + sample['label'].tolist()
    #    CONFIGURATION.log(str(ctr))
    #    if sample.plus_diff.values[0] > 0.68 and sample.label.values[0] == 1 or sample.plus_diff.values[0] < 0.68 and sample.label.values[0] == 0:
    #        plus_prediction = plus_prediction + model.predict(sample.loc[:, sample.columns != 'label']).tolist()
    #        plus_gold = plus_gold + sample['label'].tolist()
    #    CONFIGURATION.log(str(sample.iloc[0].tolist() + [prediction]) + '\n')

    prediction = np.array(prediction)
    gold = np.array(gold)
    plus_prediction = np.array(plus_prediction)
    plus_gold = np.array(plus_gold)

    result = classification_report(prediction, gold, target_names=['false', 'true'])
    CONFIGURATION.log("Results on test:")
    CONFIGURATION.log(result)
    CONFIGURATION.log(ConfusionMatrix(prediction, gold))
    CONFIGURATION.log("\n\n--------------------------------------------------------------\n")
    CONFIGURATION.log("Results on test:")
    CONFIGURATION.log(str(result))
    CONFIGURATION.log(str(ConfusionMatrix(prediction, gold)))
    CONFIGURATION.log("\n\n--------------------------------------------------------------\n")

    plus_result = classification_report(plus_prediction, plus_gold, target_names=['false', 'true'])
    CONFIGURATION.log("Results on test:")
    CONFIGURATION.log(plus_result)
    CONFIGURATION.log(ConfusionMatrix(plus_prediction, plus_gold))
    CONFIGURATION.log("\n\n--------------------------------------------------------------\n")
    CONFIGURATION.log("Results on test:")
    CONFIGURATION.log(str(plus_result))
    CONFIGURATION.log(str(ConfusionMatrix(plus_prediction, plus_gold)))
    CONFIGURATION.log("\n\n--------------------------------------------------------------\n")

    CONFIGURATION.log("Syntactic matching results+ on test: 0.0%")
    CONFIGURATION.log("Syntactic matching results+ on test: 0.0%")

    CONFIGURATION.log("\n################################################################\n\n")
    CONFIGURATION.log("\n################################################################\n\n")


    return PipelineDataTuple(graph1, graph2)


def interface(main_input, args, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    graph1 = main_input.get(0)
    graph2 = main_input.get(1)
    model = args.get(0)
    assert graph1 is not None, "Graph (1) not found in " + os.path.basename(sys.argv[0])
    assert graph2 is not None, "Graph (2) not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.gold_mapping is not None, "Path to gold standard file not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.logfile is not None, "Path to logfile not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.name is not None, "Test config name not found in " + os.path.basename(sys.argv[0])
    assert model is not None, "No model provided in " + os.path.basename(sys.argv[0])
    return exec(graph1, graph2, model)
