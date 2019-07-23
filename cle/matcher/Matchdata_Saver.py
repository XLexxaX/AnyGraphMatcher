from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os
import re
import hashlib
import pandas as pd
import sklearn
import scipy
from matcher.DatasetHelperTools import batch_prepare_data_from_graph, get_schema_data_from_graph, extend_features, \
    extract_non_trivial_matches, stream_prepare_data_from_graph
from configurations.PipelineTools import PipelineDataTuple
from StringMatching import parallel
import sys
import uuid
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
import ntpath

global CONFIGURATION

def exec(graph1, graph2):

            for gold_mapping in CONFIGURATION.gold_mapping.raw_trainsets:
                print("     --> Preparing training data.")
                # package_directory = os.path.dirname(os.path.abspath(__file__))
                #gold_mapping = CONFIGURATION.gold_mapping.raw_trainsets[0]#os.path.join(package_directory, '..','..', 'data', 'sap_hilti_data','sap_hilti_full_strings',
                               #     'train_simple_sap_hilti.csv')
                save(graph1, graph2, ntpath.basename(gold_mapping), gold_mapping)
                path_to_set = CONFIGURATION.rundir + ntpath.basename(gold_mapping) + "-strcombined.csv"
                path_to_idset = CONFIGURATION.rundir + ntpath.basename(gold_mapping) + "-strcombined_ids.csv"
                df = pd.read_csv(path_to_set, index_col=['Unnamed: 0'], sep="\t", encoding="UTF-8").\
                        merge(pd.read_csv(path_to_idset, index_col=['Unnamed: 0'], sep="\t", encoding="UTF-8"), left_index=True,\
                        right_index=True)
                df.to_csv(CONFIGURATION.rundir + ntpath.basename(gold_mapping) + "_merged.csv", sep="\t")
                CONFIGURATION.gold_mapping.prepared_trainsets.append(df)

            #gold_mapping = CONFIGURATION.gold_mapping.raw_trainsets[1] #os.path.join(package_directory, '..','..', 'data', 'sap_hilti_data','sap_hilti_full_strings',
            #                #'train_hard_sap_hilti.csv')
            #save(graph1, graph2, 'train_hard', gold_mapping)
            #CONFIGURATION.gold_mapping.prepared_trainsets.append(CONFIGURATION.rundir + 'train_hard' + "-strcombined.csv")

            if CONFIGURATION.match_cross_product:
                print("     --> No testset provided. Preparing cross product.")
                filepath = CONFIGURATION.rundir + str(uuid.uuid4().hex)+".tmp"
                print('         Blocking by syntax, progress: 0%', end="\r")
                parallel.main(CONFIGURATION.src_triples, CONFIGURATION.tgt_triples, CONFIGURATION.src_properties, filepath)
                print('         Blocking by syntax, progress: 100%')
                CONFIGURATION.gold_mapping.raw_testsets = [filepath]
            else:
                print("     --> Preparing testset.")
            for gold_mapping in CONFIGURATION.gold_mapping.raw_testsets:
                #gold_mapping = CONFIGURATION.gold_mapping.raw_testsets[0]#os.path.join(package_directory, '..','..', 'data', 'sap_hilti_data','sap_hilti_full_strings',
                               #     'test_simple_sap_hilti.csv')
                save(graph1, graph2, ntpath.basename(gold_mapping), gold_mapping)
                path_to_set = CONFIGURATION.rundir + ntpath.basename(gold_mapping) + "-strcombined.csv"
                path_to_idset = CONFIGURATION.rundir + ntpath.basename(gold_mapping) + "-strcombined_ids.csv"
                df = pd.read_csv(path_to_set, index_col=['Unnamed: 0'], sep="\t", encoding="UTF-8").\
                        merge(pd.read_csv(path_to_idset, index_col=['Unnamed: 0'], sep="\t", encoding="UTF-8"), left_index=True,\
                        right_index=True)
                df.to_csv(CONFIGURATION.rundir + ntpath.basename(gold_mapping) + "_merged.csv", sep="\t")
                CONFIGURATION.gold_mapping.prepared_testsets.append(df)


#            gold_mapping = CONFIGURATION.gold_mapping.raw_testsets[1]#os.path.join(package_directory, '..','..', 'data', 'sap_hilti_data','sap_hilti_full_strings',
            #               #     'test_hard_sap_hilti.csv')
#            save(graph1, graph2, 'test_hard', gold_mapping)
#            CONFIGURATION.gold_mapping.prepared_testsets.append(CONFIGURATION.rundir + 'test_hard' + "-strcombined.csv")


            return PipelineDataTuple(graph1, graph2)# just return the original graph data; this is assumed to be the final step in the pipeline!

def save(graph1, graph2, prefix, gold_mapping):
            cachefile_path = None
            cachefile = hashlib.sha256(bytes(re.escape(gold_mapping), encoding='UTF-8')).hexdigest() + '.cache'
            if os.path.exists(CONFIGURATION.cachedir + cachefile) and CONFIGURATION.use_cache:
                cachefile_path = CONFIGURATION.cachedir + cachefile

            if CONFIGURATION.use_streams:
                combined_samples, combined_samples_ids = stream_prepare_data_from_graph(graph1, graph2, gold_mapping, CONFIGURATION.calc_PLUS_SCORE, cachefile_path)
            else:
                #positive_samples, negative_samples,
                combined_samples, combined_samples_ids = batch_prepare_data_from_graph(graph1, graph2, gold_mapping, CONFIGURATION.src_properties, CONFIGURATION.tgt_properties, CONFIGURATION.calc_PLUS_SCORE, cachefile_path, config=CONFIGURATION)

                combined_samples.to_csv(CONFIGURATION.rundir + prefix + "-strcombined.csv", sep="\t")
                combined_samples_ids.to_csv(CONFIGURATION.rundir + prefix + "-strcombined_ids.csv", sep="\t")

                if not os.path.exists(CONFIGURATION.cachedir + cachefile):
                    if CONFIGURATION.calc_PLUS_SCORE:
                        cols = ['src_id', 'tgt_id', 'syntactic_diff', 'plus_diff']
                    else:
                        cols = ['src_id', 'tgt_id', 'syntactic_diff']
                    pd.merge(combined_samples, combined_samples_ids, left_index=True, right_index=True)[cols]\
                       .to_csv(CONFIGURATION.cachedir + cachefile)



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


#if __name__ == '__main__':
#    from sklearn.svm import LinearSVC
#    model = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#                      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#                      multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)
#    exec(None, None, model)
