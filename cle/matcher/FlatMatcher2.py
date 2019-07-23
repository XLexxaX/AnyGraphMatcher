from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os
import numpy as np
import pandas as pd
import sklearn
import scipy
from matcher.DatasetHelperTools import batch_prepare_data_from_graph, get_schema_data_from_graph, extend_features, \
    extract_non_trivial_matches
from configurations.PipelineTools import PipelineDataTuple
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

global CONFIGURATION

def exec(graph1, graph2, model):

            setsize = 1000
            # Now start prediction:


            positive_samples, negative_samples, combined_samples, combined_samples_ids = batch_prepare_data_from_graph(graph1, graph2, CONFIGURATION.gold_mapping)
            positive_samples, negative_samples, combined_samples = extend_features(positive_samples), extend_features(negative_samples), extend_features(combined_samples)
            non_trivial_matches_ids = extract_non_trivial_matches(graph1, graph2, combined_samples_ids, CONFIGURATION.src_properties, CONFIGURATION.tgt_properties, combined_samples)

            combined_samples.to_csv(CONFIGURATION.rundir+"combined.csv")
            combined_samples.to_csv(CONFIGURATION.projectdir+"combined.csv")
            combined_samples_ids.to_csv(CONFIGURATION.rundir+"combined_ids.csv")
            negative_samples.to_csv(CONFIGURATION.rundir+"negatives.csv")
            positive_samples.to_csv(CONFIGURATION.rundir+"positives.csv")
            package_directory = os.path.dirname(os.path.abspath(__file__))

            CONFIGURATION.log("\n\n")
            CONFIGURATION.log("#####################################################\n")
            CONFIGURATION.log("#" + CONFIGURATION.name + " / " + str(model) + "\n")
            CONFIGURATION.log("-----------------------------------------------------\n")


            #Train/Test split
            X = pd.DataFrame(combined_samples.loc[:,combined_samples.columns != 'label'])
            X = pd.concat([X, combined_samples_ids], axis=1, sort=False)
            Y = pd.DataFrame(combined_samples.loc[:,'label'])

            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.5,
                                                                                        random_state=7)


            X_test.set_index(['src_id','tgt_id'])
            combined_samples_ids.set_index(['src_id','tgt_id'])
            non_trivial_matches_ids.set_index(['src_id','tgt_id'])
            combined_samples_ids = X_test.merge(combined_samples_ids, how='inner')
            non_trivial_matches_ids = X_test.merge(non_trivial_matches_ids, how='inner')
            combined_samples_ids = combined_samples_ids.reset_index(drop = True)
            non_trivial_matches_ids = non_trivial_matches_ids.reset_index(drop = True)

            X_train = X_train.drop(['src_id','tgt_id'], axis=1)
            X_test = X_test.drop(['src_id','tgt_id'], axis=1)


            # fit model to training data
            model.fit(X_train, y_train.values.ravel())



            #scaler = StandardScaler()

            # Fit only to the training data
            #scaler.fit(X_train)



            # Now apply the transformations to the data:
            #X_train = scaler.transform(X_train)
            #X_test = scaler.transform(X_test)


            y_pred = model.predict(X_train)
            y_pred = np.array(y_pred)#scipy.stats.zscore(np.array(y_pred))
            predictions = [1 if value > 0.5 else 0 for value in y_pred]
            # evaluate predictions
            CONFIGURATION.log("Macro train: "+str(precision_recall_fscore_support(y_train, predictions, average='macro')) + "\n")
            CONFIGURATION.log("Micro train: "+str(precision_recall_fscore_support(y_train, predictions, average='micro')) + "\n")
            CONFIGURATION.log("#####################################################\n")


            y_test = y_test.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)

            y_pred = model.predict(X_test)
            y_pred = np.array(y_pred)
            #y_pred = scipy.stats.zscore(np.array(y_pred))
            persisted_predictions = [1 if value > 0.5 else 0 for value in y_pred]
            # evaluate predictions
            CONFIGURATION.log("Macro test: "+str(precision_recall_fscore_support(y_test, persisted_predictions, average='macro')) + "\n")
            CONFIGURATION.log("Micro test: "+str(precision_recall_fscore_support(y_test, persisted_predictions, average='micro')) + "\n")
            CONFIGURATION.log("#####################################################\n")
            target_names = ['neg', 'pos']
            CONFIGURATION.log("Report (pos: "+str(setsize)+" / neg: "+str(setsize)+"):\n")
            CONFIGURATION.log(str(classification_report(y_test, persisted_predictions, target_names=target_names)) + "\n")
            non_trivials = pd.merge(non_trivial_matches_ids, combined_samples_ids, left_on=['src_id','tgt_id'], right_on=['src_id','tgt_id'], how='right', indicator=True)
            non_trivials = non_trivials.loc[non_trivials['_merge'] == 'both'].index.tolist()
            #y_test = y_test['label']
            CONFIGURATION.log("#####################################################\n")
            CONFIGURATION.log("Report+ :" + str(classification_report(y_test.loc[y_test.index.isin(non_trivials)], np.array(persisted_predictions)[non_trivials], target_names=target_names)) + "\n")

            # Schema correspondence predictions
            # In the following code segment, schema correspondences are predicted using the instance-matching model.
            # However, this method is not recommended, as the model is (most likely) primarily or only trained on
            # instance-correspondences.
            '''schema_data, schema_data_ids = get_schema_data_from_graph(graph1, graph2)
            schema_data = extend_features(schema_data)
            y_pred = model.predict(schema_data)
            y_pred = scipy.stats.zscore(np.array(y_pred))
            predictions = [1 if value > 0 else 0 for value in y_pred]
            schema_predicted = pd.concat([pd.DataFrame({"prediction":predictions}), schema_data_ids], axis=1, sort=False)
            schema_predicted.to_csv(index=False,path_or_buf=package_directory+"/../../predicted_data.csv", header=False)
            pd.options.display.max_colwidth = 100
            pd.set_option('display.max_colwidth', -1)
            CONFIGURATION.log("\nschema matches predicted with ML model:\n")
            schema_predicted = schema_predicted[schema_predicted['prediction'] == 1]
            #CONFIGURATION.log(schema_predicted.to_string()+"\n")'''


            CONFIGURATION.log("\nschema matches predicted with heuristics:\n")
            persisted_predictions = [x == 1 for x in persisted_predictions]
            positive_predictions = combined_samples_ids[persisted_predictions]
            correspondece_types = dict()
            for index, row in positive_predictions.iterrows():
                try:
                    srckey = str(graph1.elements[row['src_id']].relations['http://www.w3.org/1999/02/22-rdf-syntax-ns#type'].descriptor)
                    tgtkey = str(graph2.elements[row['tgt_id']].relations['http://www.w3.org/1999/02/22-rdf-syntax-ns#type'].descriptor)
                    if (srckey in correspondece_types.keys()):
                        if (tgtkey in correspondece_types[srckey].keys()):
                            correspondece_types[srckey][tgtkey] = correspondece_types[srckey][tgtkey] + 1
                        else:
                            correspondece_types[srckey][tgtkey] = 1
                    else:
                        correspondece_types[srckey] = dict()
                        correspondece_types[srckey][tgtkey] = 1
                except:
                    pass

            for srckey, val in correspondece_types.items():
                maxtgtkey = None
                for tgtkey, count in val.items():
                    if maxtgtkey == None:
                        maxtgtkey = tgtkey
                    if count > val[maxtgtkey]:
                        maxtgtkey = tgtkey
                CONFIGURATION.log(str(srckey) + " --> " + str(maxtgtkey) + "\n")

            CONFIGURATION.log("\n\n\n")
            print("     --> Evaluated; logs written to " + str(CONFIGURATION.logfile))

            return PipelineDataTuple(graph1, graph2)# just return the original graph data; this is assumed to be the final step in the pipeline!

def interface(main_input, args, configuration):
    global CONFIGURATION
    CONFIGURATION = configuration
    graph1 = main_input.get(0)
    graph2 = main_input.get(1)
    model = args.get(0)
    assert graph1 is not None, "Graph (1) not found in " + os.path.basename(sys.argv[0])
    assert graph2 is not None, "Graph (2) not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.gold_mapping is not None, "Path to gold standard file not found in " + os.path.basename(sys.argv[0])
    assert model is not None, "ML model not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.logfile is not None, "Path to logfile not found in " + os.path.basename(sys.argv[0])
    assert CONFIGURATION.name is not None, "Test config name not found in " + os.path.basename(sys.argv[0])
    return exec(graph1, graph2, model)


#if __name__ == '__main__':
#    from sklearn.svm import LinearSVC
#    model = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#                      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#                      multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)
#    exec(None, None, model)
