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
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate

global CONFIGURATION

def exec(graph1, graph2, model):

            setsize = 1000
            # Now start prediction:


            package_directory = os.path.dirname(os.path.abspath(__file__))
            CONFIGURATION.gold_mapping = os.path.join(package_directory, '..','..', 'data', 'sap_hilti_data', 'sap_hilti_full_strings',
                                'hq_sap_hilti_gold_sampled.csv')
            positive_samples, negative_samples, combined_samples, combined_samples_ids = batch_prepare_data_from_graph(graph1, graph2, CONFIGURATION.gold_mapping)
            positive_samples, negative_samples, combined_samples = extend_features(positive_samples), extend_features(negative_samples), extend_features(combined_samples)
            non_trivial_matches_ids = extract_non_trivial_matches(graph1, graph2, combined_samples_ids, CONFIGURATION.src_properties, CONFIGURATION.tgt_properties, combined_samples)

            combined_samples.to_csv(CONFIGURATION.rundir+"scombined.csv")
            #pd.merge(pd.merge(non_trivial_matches_ids, combined_samples_ids, left_on=['src_id','tgt_id'], right_on=['src_id','tgt_id'], how='inner', indicator=False),
            #         combined_samples, right_index=True, left_index=True).drop(['src_id','tgt_id'], axis=1).to_csv(CONFIGURATION.rundir+"snon_trivials.csv")
            combined_samples_ids.to_csv(CONFIGURATION.rundir+"scombined_ids.csv")
            negative_samples.to_csv(CONFIGURATION.rundir+"snegatives.csv")
            positive_samples.to_csv(CONFIGURATION.rundir+"spositives.csv")

            CONFIGURATION.log("\n\n")
            CONFIGURATION.log("#####################################################\n")
            CONFIGURATION.log("#" + CONFIGURATION.name + " / " + str(model) + "\n")
            CONFIGURATION.log("-----------------------------------------------------\n")


            #Train/Test split
            X = pd.DataFrame(combined_samples.loc[:,combined_samples.columns != 'label'])
            #X = pd.concat([X, combined_samples_ids], axis=1, sort=False)
            Y = pd.DataFrame(combined_samples.loc[:,'label'])


            from sklearn import metrics
            cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
            per = cross_validate(model, X, Y, cv=cv,scoring=('f1_micro', 'f1_macro', 'precision','recall'), return_train_score = True)
            CONFIGURATION.log("F1-macro test\t" + str(np.average(per['test_f1_macro'])) + " +/-" + str(np.std(per['test_f1_macro'])) + "\t" + str(per['test_f1_macro']) + "\n")
            CONFIGURATION.log("F1-macro train\t" + str(np.average(per['train_f1_macro'])) + " +/-" + str(np.std(per['train_f1_macro'])) + "\t" + str(
                per['train_f1_macro']) + "\n")
            CONFIGURATION.log("F1-micro test:\t" + str(np.average(per['test_f1_micro'])) + " +/-" + str(np.std(per['test_f1_micro'])) + "\t" + str(per['test_f1_micro']) + "\n")
            CONFIGURATION.log("F1-micro train:\t" + str(np.average(per['train_f1_micro'])) + " +/-" + str(np.std(per['train_f1_micro'])) + "\t" + str(
                per['train_f1_micro']) + "\n")
            CONFIGURATION.log("Precision test:\t" + str(np.average(per['test_precision'])) + " +/-" + str(np.std(per['test_precision'])) + "\t" + str(per['test_precision']) + "\n")
            CONFIGURATION.log("Precision train:\t" + str(np.average(per['train_precision'])) + " +/-" + str(np.std(per['train_precision'])) + "\t" + str(
                per['train_precision']) + "\n")
            CONFIGURATION.log("Recall test:\t\t" + str(np.average(per['test_recall'])) + " +/-" + str(np.std(per['test_recall'])) + "\t" + str(per['test_recall']) + "\n")
            CONFIGURATION.log("Recall train:\t\t" + str(np.average(per['train_recall'])) + " +/-" + str(np.std(per['train_recall'])) + "\t" + str(
                per['train_recall']) + "\n")
            from sklearn.model_selection import cross_val_predict
            y_pred = cross_val_predict(model, X, Y, cv=cv)
            y_pred = np.array(y_pred)#scipy.stats.zscore(np.array(y_pred))
            predictions = [1 if value > 0.5 else 0 for value in y_pred]
            # evaluate predictions


            persisted_predictions = [1 if value > 0.5 else 0 for value in y_pred]

            CONFIGURATION.log('\nDataset meta info:\n')
            CONFIGURATION.log('Actual samples ' + str(len(Y)) + ' / Positive samples ' + str(len(Y.loc[Y['label']==1])) + ' / Negative samples ' + str(len(Y.loc[Y['label']==0])) + '\n')
            CONFIGURATION.log('Predicted samples ' + str(len(Y)) + ' / Positive samples ' + str(len(np.where(np.array(persisted_predictions)==1)[0])) + ' / Negative samples ' + str(len(np.where(np.array(persisted_predictions)==0)[0])) + '\n')
            CONFIGURATION.log("#####################################################\n")


            # evaluate predictions
            non_trivials = pd.merge(non_trivial_matches_ids, combined_samples_ids, left_on=['src_id','tgt_id'], right_on=['src_id','tgt_id'], how='right', indicator=True)
            non_trivials = non_trivials.loc[non_trivials['_merge'] == 'both'].index.tolist()
            #y_test = y_test['label']
            target_names=['pos','neg']
            CONFIGURATION.log("Report+:" + str(
                classification_report(Y.loc[Y.index.isin(non_trivials)], np.array(persisted_predictions)[non_trivials],
                                      target_names=target_names)) + "\n")
            CONFIGURATION.log('\nDataset meta info:\n')
            CONFIGURATION.log('Actual samples ' + str(len(non_trivials)) + ' / Positive samples ' + str(
                len(np.where(np.array(persisted_predictions)[non_trivials] == 1)[0])) + ' / Negative samples ' + str(
                len(np.where(np.array(persisted_predictions)[non_trivials] == 0)[0])) + '\n')
            CONFIGURATION.log('Predicted samples ' + str(len(non_trivials)) + ' / Positive samples ' + str(
                len(np.where(np.array(Y.loc[Y.index.isin(non_trivials)]) == 1)[0])) + ' / Negative samples ' + str(
                len(np.where(np.array(Y.loc[Y.index.isin(non_trivials)]) == 0)[0])) + '\n')

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
            CONFIGURATION.log("     --> Evaluated; logs written to " + str(CONFIGURATION.logfile))

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
