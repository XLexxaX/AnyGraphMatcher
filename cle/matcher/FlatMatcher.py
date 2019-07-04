import pandas as pd
import numpy as np
import ntpath

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from pandas_ml import ConfusionMatrix

from cle.configurations.PipelineTools import PipelineDataTuple
from cle.matcher import Matchdata_Saver, PredictionToXMLConverter
import sys
import os
from joblib import dump, load


global CONFIGURATION

def exec(graph1, graph2, ml_model):

    Matchdata_Saver.interface(PipelineDataTuple(graph1, graph2), None, CONFIGURATION)

    #train_simple = pd.read_csv(CONFIGURATION.rundir + 'train_simple-strcombined.csv', index_col=['Unnamed: 0'])
    #train_hard = pd.read_csv(CONFIGURATION.rundir + 'train_hard-strcombined.csv', index_col=['Unnamed: 0'])
    #test_simple = pd.read_csv(CONFIGURATION.rundir + 'test_simple-strcombined.csv', index_col=['Unnamed: 0'])
    #test_hard = pd.read_csv(CONFIGURATION.rundir + 'test_hard-strcombined.csv', index_col=['Unnamed: 0'])

    train = None
    for trainset in CONFIGURATION.gold_mapping.prepared_trainsets:
        if train is None:
            train = trainset.loc[:, ~(trainset.columns.isin(['src_id','tgt_id', 'src_category', 'tgt_category']))]#pd.read_csv(trainset, index_col=['Unnamed: 0'])
        else:
            tmp_train = trainset.loc[:, ~(trainset.columns.isin(['src_id','tgt_id', 'src_category', 'tgt_category']))] #pd.read_csv(trainset, index_col=['Unnamed: 0'])
            train = train.append(tmp_train, ignore_index=True)

    # #### Alternative 1: Sample the training data manually.
    #a = train_simple.loc[train_simple['label']==1].sample(n=100, replace=False)
    #b = train_simple.loc[train_simple['label']==0].sample(n=100, replace=False)
    #c = train_hard.loc[train_hard['label']==1].sample(n=0, replace=False)
    #d = train_hard.loc[train_hard['label']==0].sample(n=600, replace=False)
    #train = d.append(c.append(a.append(b, ignore_index=True), ignore_index=True), ignore_index=True)


    # #### Alternative 2: Use all available data for training.
    #train = train_simple.append(train_hard, ignore_index=True)

    # ## Prepare train/test/prediction data
    x_train = train.loc[:, train.columns != 'label']
    y_train = train['label']


    # ## Prediction
    model = ml_model  # RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0) #LogisticRegression(solver='lbfgs')
    model = model.fit(
        x_train[[col for col in x_train.columns]],
        y_train)
    syntactic_model = LogisticRegression(
        solver='lbfgs')  # RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    syntactic_model = syntactic_model.fit(pd.DataFrame(x_train['syntactic_diff']), y_train)
    dump(model, CONFIGURATION.rundir + 'model.joblib')
    dump(model, CONFIGURATION.rundir + 'syntactic_model.joblib')

    for testset in CONFIGURATION.gold_mapping.prepared_testsets:
        test = testset.loc[:, ~(testset.columns.isin(['src_id','tgt_id', 'src_category', 'tgt_category']))]#pd.read_csv(testset, index_col=['Unnamed: 0'])

        x_test1 = test.loc[:, test.columns != 'label']
        y_test1 = test['label']

        CONFIGURATION.log("\n################################################################\n\n")

        prediction = model.predict(x_test1[[col for col in x_test1.columns]])
        result = classification_report(np.array(y_test1), prediction, target_names=['false','true'])
        CONFIGURATION.log("FlatMatcher - ml_model performance:\n")
        CONFIGURATION.log(str(result))
        CONFIGURATION.log(str(ConfusionMatrix(np.array(y_test1), prediction)))
        CONFIGURATION.log("\n\n--------------------------------------------------------------\n")

        CONFIGURATION.log("\n" + str([col for col in x_test1.columns]))
        CONFIGURATION.log("\n" + str(LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(x_train[[col for col in x_train.columns]], y_train).coef_) + "\n")

        testset.loc[prediction==1, ['src_id','tgt_id']].to_csv(CONFIGURATION.rundir + 'ml_matchings.csv', sep="\t", index=False, encoding='UTF-8')

        PredictionToXMLConverter.interface(PipelineDataTuple(graph1, graph2), PipelineDataTuple('ml_matchings.csv'), CONFIGURATION)


        #prediction = syntactic_model.predict(pd.DataFrame(x_test1['syntactic_diff']))
        #result = classification_report(prediction, np.array(y_test1), target_names=['false','true'])
        #print("Syntactic matching results on simple test:")
        #print(result)
        #print(ConfusionMatrix(prediction, np.array(y_test1)))
        #CONFIGURATION.log("Syntactic matching results on simple test:")
        #CONFIGURATION.log(str(result))
        #CONFIGURATION.log(str(ConfusionMatrix(prediction, np.array(y_test1))))
    #
        CONFIGURATION.log("\n################################################################\n\n")


        if CONFIGURATION.calc_PLUS_SCORE:
            test_plus = test.loc[(test.plus_diff>0.68) & (test.label==1)]
            test_plus = test_plus.append(test.loc[(test.plus_diff<0.68) & (test.label==0)], ignore_index=True)
            x_test_plus = test_plus.loc[:, test_plus.columns != 'label']
            y_test_plus = test_plus['label']
            prediction_plus = model.predict(x_test_plus[[col for col in x_train.columns]])
            result_plus = classification_report(np.array(y_test_plus), prediction_plus, target_names=['false','true'])
            CONFIGURATION.log("FlatMatcher - ml_model performance+:\n")
            CONFIGURATION.log(str(result_plus))
            CONFIGURATION.log(str(ConfusionMatrix(np.array(y_test_plus), prediction_plus)))
            CONFIGURATION.log("\n\n--------------------------------------------------------------\n")
        else:
            CONFIGURATION.log("No performance+ calculated")
            CONFIGURATION.log("\n\n--------------------------------------------------------------\n")
        #print("Syntactic matching results+ on simple test: 0.0%")
        #CONFIGURATION.log("Syntactic matching results+ on simple test: 0.0%")

        CONFIGURATION.log("\n################################################################\n\n")


    # Schema correspondence predictions
    # In the following code segment, schema correspondences are predicted using the instance-matching model.
    # However, this method is not recommended, as the model is (most likely) primarily or only trained on
    # instance-correspondences.
    '''import scipy
    from cle.matcher.DatasetHelperTools import extend_features, get_schema_data_from_graph
    schema_data, schema_data_ids = get_schema_data_from_graph(graph1, graph2)
    schema_data = extend_features(schema_data)
    y_pred = model.predict(schema_data)
    y_pred = scipy.stats.zscore(np.array(y_pred))
    predictions = [1 if value > 0 else 0 for value in y_pred]
    schema_predicted = pd.concat([pd.DataFrame({"prediction":predictions}), schema_data_ids], axis=1, sort=False)
    schema_predicted.to_csv(index=False,path_or_buf=CONFIGURATION.rundir+"predicted_data.csv", header=False)
    pd.options.display.max_colwidth = 100
    pd.set_option('display.max_colwidth', -1)
    CONFIGURATION.log("\nschema matches predicted with ML model:\n")
    schema_predicted = schema_predicted[schema_predicted['prediction'] == 0]
    CONFIGURATION.log(schema_predicted.to_string()+"\n")'''

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
