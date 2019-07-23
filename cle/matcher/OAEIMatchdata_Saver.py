from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os
import hashlib
import re
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

def exec(graph1, graph2):

            gold_mapping = CONFIGURATION.gold_mapping.raw_trainsets[0]
            save(graph1, graph2, 'train', gold_mapping)
            CONFIGURATION.gold_mapping.prepared_trainsets.append(CONFIGURATION.rundir + 'train' + "-strcombined.csv")

            return PipelineDataTuple(graph1, graph2)# just return the original graph data; this is assumed to be the final step in the pipeline!

def save(graph1, graph2, prefix, gold_mapping):
            cachefile_path = None
            cachefile = hashlib.sha256(bytes(re.escape(gold_mapping), encoding='UTF-8')).hexdigest() + '.cache'
            if os.path.exists(CONFIGURATION.cachedir + cachefile) and CONFIGURATION.use_cache:
                cachefile_path = CONFIGURATION.cachedir + cachefile

            positive_samples, negative_samples, combined_samples, combined_samples_ids = batch_prepare_data_from_graph(graph1, graph2, gold_mapping, cachefile_path)
            positive_samples, negative_samples, combined_samples = extend_features(positive_samples), extend_features(negative_samples), extend_features(combined_samples)

            combined_samples.to_csv(CONFIGURATION.rundir + prefix + "-strcombined.csv")
            combined_samples_ids.to_csv(CONFIGURATION.rundir + prefix + "-strcombined_ids.csv")

            if not os.path.exists(CONFIGURATION.cachedir + cachefile):
                pd.merge(combined_samples, combined_samples_ids, left_index=True, right_index=True)[['src_id', 'tgt_id', 'syntactic_diff', 'plus_diff']]\
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
