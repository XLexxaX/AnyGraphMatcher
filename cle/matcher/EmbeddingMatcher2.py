import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from pandas_ml import ConfusionMatrix

from cle.configurations.PipelineTools import PipelineDataTuple
from cle.matcher import Matchdata_Saver
import sys
import os
from joblib import dump, load


global CONFIGURATION

def exec(graph1, graph2, ml_model):

    Matchdata_Saver.interface(PipelineDataTuple(graph1, graph2), None, CONFIGURATION)

    train_simple = pd.read_csv(CONFIGURATION.rundir + 'train_simple-strcombined.csv', index_col=['Unnamed: 0'])
    train_hard = pd.read_csv(CONFIGURATION.rundir + 'train_hard-strcombined.csv', index_col=['Unnamed: 0'])
    test_simple = pd.read_csv(CONFIGURATION.rundir + 'test_simple-strcombined.csv', index_col=['Unnamed: 0'])
    test_hard = pd.read_csv(CONFIGURATION.rundir + 'test_hard-strcombined.csv', index_col=['Unnamed: 0'])



    # #### Alternative 1: Sample the training data manually.
    #a = train_simple.loc[train_simple['label']==1].sample(n=100, replace=False)
    #b = train_simple.loc[train_simple['label']==0].sample(n=100, replace=False)
    #c = train_hard.loc[train_hard['label']==1].sample(n=0, replace=False)
    #d = train_hard.loc[train_hard['label']==0].sample(n=600, replace=False)
    #train = d.append(c.append(a.append(b, ignore_index=True), ignore_index=True), ignore_index=True)


    # #### Alternative 2: Use all available data for training.
    train = train_simple.append(train_hard, ignore_index=True)


    # ## Prepare train/test/prediction data
    x_train = train.loc[:, train.columns != 'label']
    y_train = train['label']

    x_test1 = test_simple.loc[:, test_simple.columns != 'label']
    y_test1 = test_simple['label']

    x_test2 = test_hard.loc[:, test_hard.columns != 'label']
    y_test2 = test_hard['label']


    # ## Prediction
    model = ml_model#RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0) #LogisticRegression(solver='lbfgs')
    model = model.fit(x_train[[col for col in x_train.columns if not col == 'syntactic_diff' and not col == 'plus_diff']], y_train)
    syntactic_model = LogisticRegression(solver='lbfgs')#RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    syntactic_model = syntactic_model.fit(pd.DataFrame(x_train['syntactic_diff']), y_train)
    dump(model, CONFIGURATION.rundir + 'model.joblib')
    dump(model, CONFIGURATION.rundir + 'syntactic_model.joblib')




    print("\n################################################################\n\n")
    CONFIGURATION.log("\n################################################################\n\n")

    prediction = model.predict(x_test1[[col for col in x_train.columns if not col == 'syntactic_diff' and not col == 'plus_diff']])
    result = classification_report(prediction, np.array(y_test1), target_names=['false','true'])
    print("Results on simple test:")
    print(result)
    print(ConfusionMatrix(prediction, np.array(y_test1)))
    print("\n\n--------------------------------------------------------------\n")
    CONFIGURATION.log("Results on simple test:")
    CONFIGURATION.log(str(result))
    CONFIGURATION.log(str(ConfusionMatrix(prediction, np.array(y_test1))))
    CONFIGURATION.log("\n\n--------------------------------------------------------------\n")

    prediction = syntactic_model.predict(pd.DataFrame(x_test1['syntactic_diff']))
    result = classification_report(prediction, np.array(y_test1), target_names=['false','true'])
    print("Syntactic matching results on simple test:")
    print(result)
    print(ConfusionMatrix(prediction, np.array(y_test1)))
    CONFIGURATION.log("Syntactic matching results on simple test:")
    CONFIGURATION.log(str(result))
    CONFIGURATION.log(str(ConfusionMatrix(prediction, np.array(y_test1))))

    print("\n################################################################\n\n")
    CONFIGURATION.log("\n################################################################\n\n")



    test_plus = test_simple.loc[(test_simple.plus_diff>0.68) & (test_simple.label==1)]
    test_plus = test_plus.append(test_simple.loc[(test_simple.plus_diff<0.68) & (test_simple.label==0)], ignore_index=True)
    x_test_plus = test_plus.loc[:, test_plus.columns != 'label']
    y_test_plus = test_plus['label']
    prediction_plus = model.predict(x_test_plus[[col for col in x_train.columns if not col == 'syntactic_diff' and not col == 'plus_diff']])
    result_plus = classification_report(prediction_plus, np.array(y_test_plus), target_names=['false','true'])
    print("Results+ on simple test:")
    print(result_plus)
    print(ConfusionMatrix(prediction_plus, np.array(y_test_plus)))
    print("\n\n--------------------------------------------------------------\n")
    CONFIGURATION.log("Results+ on simple test:")
    CONFIGURATION.log(str(result_plus))
    CONFIGURATION.log(str(ConfusionMatrix(prediction_plus, np.array(y_test_plus))))
    CONFIGURATION.log("\n\n--------------------------------------------------------------\n")

    print("Syntactic matching results+ on simple test: 0.0%")
    CONFIGURATION.log("Syntactic matching results+ on simple test: 0.0%")

    print("\n################################################################\n\n")
    CONFIGURATION.log("\n################################################################\n\n")



    prediction2 = model.predict(x_test2[[col for col in x_train.columns if not col == 'syntactic_diff' and not col == 'plus_diff']])
    result2 = classification_report(prediction2, np.array(y_test2), target_names=['false','true'])
    print("Results on hard test:")
    print(str(result2))
    print(str(ConfusionMatrix(prediction2, np.array(y_test2))))
    print("\n\n--------------------------------------------------------------\n")
    CONFIGURATION.log("Results on hard test:")
    CONFIGURATION.log(str(result2))
    CONFIGURATION.log(str(ConfusionMatrix(prediction2, np.array(y_test2))))
    CONFIGURATION.log("\n\n--------------------------------------------------------------\n")

    prediction = syntactic_model.predict(pd.DataFrame(x_test2['syntactic_diff']))
    result = classification_report(prediction, np.array(y_test2), target_names=['false','true'])
    print("Syntactic matching results on hard test:")
    print(result)
    print(ConfusionMatrix(prediction, np.array(y_test2)))
    CONFIGURATION.log("Syntactic matching results on hard test:")
    CONFIGURATION.log(str(result))
    CONFIGURATION.log(str(ConfusionMatrix(prediction, np.array(y_test2))))

    print("\n################################################################\n\n")
    CONFIGURATION.log("\n################################################################\n\n")



    test_plus = test_hard.loc[(test_hard.plus_diff>0.68) & (test_hard.label==1)]
    test_plus = test_plus.append(test_hard.loc[(test_hard.plus_diff<0.68) & (test_hard.label==0)], ignore_index=True)
    x_test_plus = test_plus.loc[:, test_plus.columns != 'label']
    y_test_plus = test_plus['label']
    prediction_plus = model.predict(x_test_plus[[col for col in x_train.columns if not col == 'syntactic_diff' and not col == 'plus_diff']])
    result_plus = classification_report(prediction_plus, np.array(y_test_plus), target_names=['false','true'])
    print("Results+ on hard test:")
    print(result_plus)
    print(ConfusionMatrix(prediction_plus, np.array(y_test_plus)))
    print("\n\n--------------------------------------------------------------\n")
    CONFIGURATION.log("Results+ on hard test:")
    CONFIGURATION.log(str(result_plus))
    CONFIGURATION.log(str(ConfusionMatrix(prediction_plus, np.array(y_test_plus))))
    CONFIGURATION.log("\n\n--------------------------------------------------------------\n")

    print("Syntactic matching results+ on hard test: 0.0%")
    CONFIGURATION.log("Syntactic matching results+ on hard test: 0.0%")

    print("\n################################################################\n\n")
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
