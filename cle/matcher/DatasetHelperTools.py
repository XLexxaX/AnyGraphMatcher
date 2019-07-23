import pandas as pd
import warnings
import numpy as np
import os
import random
import re
from sklearn.utils import shuffle
from graphdatatools.InvertedIndexToolbox import getNGrams, InvertedIndex
import editdistance
from matcher import DatasetPostHelperTools

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


global syntaxprogress
syntaxprogress = 0

def warn(*args, **kwargs):
    pass
warnings.warn = warn


def get_schema_data_from_graph(graph1, graph2):
    pd.options.mode.chained_assignment = None

    print("     --> Preparing data for matching ...")

    # extract the relevant data from the graph
    tmp = list()
    for descriptor, resource in graph1.elements.items():
        for relation_descriptor, relation_resource in resource.relations.items():
            if relation_resource.descriptor == 'http://www.w3.org/2000/01/rdf-schema#class' or \
                    relation_resource.descriptor == 'http://www.w3.org/2000/01/rdf-schema#property':
                tmp.append([descriptor] + list(resource.embeddings[0]))
    src = pd.DataFrame(tmp)

    tmp = list()
    for descriptor, resource in graph2.elements.items():
        for relation_descriptor, relation_resource in resource.relations.items():
            if relation_resource.descriptor == 'http://www.w3.org/2000/01/rdf-schema#class' or \
                    relation_resource.descriptor == 'http://www.w3.org/2000/01/rdf-schema#property':
                tmp.append([descriptor] + list(resource.embeddings[0]))
    tgt = pd.DataFrame(tmp)

    src = src.dropna(axis=1)
    tgt = tgt.dropna(axis=1)
    src.columns = ["src_id"] + ["src_" + str(i) for i in range(len(src.columns) - 1)]
    tgt.columns = ["tgt_id"] + ["tgt_" + str(i) for i in range(len(tgt.columns) - 1)]

    src['cartesianproductkey'] = 0
    tgt['cartesianproductkey'] = 0
    cartesian_product = pd.merge(src, tgt, on='cartesianproductkey', how='inner')
    cartesian_product = cartesian_product.drop(['cartesianproductkey'], axis=1)

    # Ruleset - Rule 1: Only match resources, which describe similar types in the ontology.
    # E.g. classes only matched to classes, properties only to properties, ....
    for index, row in cartesian_product.iterrows():
        if not type(graph1.elements[row.src_id].type) == type(graph2.elements[row.tgt_id].type):
            cartesian_product = cartesian_product[cartesian_product.index != index]

    return cartesian_product[[col for col in cartesian_product.columns if col not in ["src_id","tgt_id"]]], cartesian_product[["src_id","tgt_id"]]

def batch_prepare_data_from_graph(graph1, graph2, gold_path, src_properties=None, tgt_properties=None, calc_PLUS_SCORE=True, syntactic_cache_file = None, n_samples=100000, config=None):

    print("         Reading embeddings, progress: 0%", end='\r')

    gold_mapping = pd.read_csv(gold_path, delimiter="\t", header=None, skiprows=1)
    gold_mapping = gold_mapping.applymap(lambda s:s.lower() if type(s) == str else s)
    gold_mapping.columns = ["gold_src_id", "gold_tgt_id", "label"]
    # Sample the data according to the parameters given.

    if gold_mapping.shape[0] < n_samples:
        n_samples = gold_mapping.shape[0]


    if syntactic_cache_file is not None:
        syntactic_cache_file = pd.read_csv(syntactic_cache_file, index_col=['Unnamed: 0'])
        syntactic_cache_file.columns = ['src_id2', 'tgt_id2', 'syntactic_diff', 'plus_diff']

    gold_mapping = gold_mapping.sample(n=n_samples)

    combined_samples = None
    for index, row in gold_mapping.iterrows():
        src_embeddings = graph1.elements[row['gold_src_id']].embeddings[0]
        tgt_embeddings = graph2.elements[row['gold_tgt_id']].embeddings[0]
        if combined_samples is None:
            src_columns = ["src_id"] + ["src_" + str(i) for i in range(len(src_embeddings))]
            tgt_columns = ["tgt_id"] + ["tgt_" + str(i) for i in range(len(tgt_embeddings))]
            combined_samples = pd.DataFrame([[row['gold_src_id']] + src_embeddings + [row['gold_tgt_id']] + tgt_embeddings + [row['label']]])
            combined_samples.columns = src_columns + tgt_columns + ['label']
        else:
            tmp = pd.DataFrame([[row['gold_src_id']] + src_embeddings + [row['gold_tgt_id']] + tgt_embeddings + [row['label']]])
            tmp.columns = src_columns + tgt_columns + ['label']
            combined_samples = combined_samples.append(tmp, ignore_index=True)

    #positive_samples = combined_samples.loc[combined_samples.label == 1]
    #negative_samples = combined_samples.loc[combined_samples.label == 0]


    print("         Reading embeddings, progress: 100%")


    # Add syntactic similarity scores.
    global syntaxprogress
    syntaxprogress = 0
    if syntactic_cache_file is None:
        combined_samples = compute_append_syntactic_similarity_score(graph1, graph2, combined_samples, 'jaccard', src_properties+tgt_properties)#["http://rdata2graph.sap.com/hilti_erp/property/mara_fert.maktx", "http://rdata2graph.sap.com/hilti_web/property/products.name"]
        #negative_samples = compute_append_syntactic_similarity_score(graph1, graph2, negative_samples, 'jaccard')
        #positive_samples = compute_append_syntactic_similarity_score(graph1, graph2, positive_samples, 'jaccard')
        if calc_PLUS_SCORE:
            combined_samples = compute_append_syntactic_similarity_score(graph1, graph2, combined_samples, 'levenshtein', src_properties+tgt_properties, True)
        #negative_samples = compute_append_syntactic_similarity_score(graph1, graph2, negative_samples, 'levenshtein', None, True)
        #positive_samples = compute_append_syntactic_similarity_score(graph1, graph2, positive_samples, 'levenshtein', None, True)
    else:
        combined_samples = append_syntactic_similarity_score(combined_samples, syntactic_cache_file)
        #negative_samples = append_syntactic_similarity_score(negative_samples, syntactic_cache_file)
        #positive_samples = append_syntactic_similarity_score(positive_samples, syntactic_cache_file)
    syntaxprogress = 0



    print("         Computing implicit embedding-properties, progress: 0%", end='\r')
    #positive_samples, negative_samples, combined_samples = extend_features(positive_samples), extend_features(negative_samples), extend_features(combined_samples)
    combined_samples = extend_features(combined_samples)
    print("         Computing implicit embedding-properties, progress: 100%")

    combined_samples = DatasetPostHelperTools.exec(combined_samples, config, graph1, graph2)


    combined_samples_ids = combined_samples[['src_id', 'tgt_id']]
    combined_samples = combined_samples.drop(['src_id', 'tgt_id'], axis=1)
    #positive_samples = positive_samples.drop(['src_id', 'tgt_id', 'label'], axis=1)
    #negative_samples = negative_samples.drop(['src_id', 'tgt_id', 'label'], axis=1)

    combined_samples_ids = combined_samples_ids.reset_index(drop=True)
    #positive_samples = positive_samples.reset_index(drop=True)
    #negative_samples = negative_samples.reset_index(drop=True)
    combined_samples = combined_samples.reset_index(drop=True)


    return combined_samples, combined_samples_ids

# For purposes of simplicity, we use pandas for each sample
def stream_prepare_data_from_graph(graph1, graph2, gold_mapping_path, calc_PLUS_SCORE=True, syntactic_cache_file=None):
    gold_mapping = pd.read_csv(gold_mapping_path, sep='\t', header=None, names=['src_id', 'tgt_id', 'label'])
    if syntactic_cache_file is not None:
        syntactic_cache_file = pd.read_csv(syntactic_cache_file, index_col=['Unnamed: 0'])
        syntactic_cache_file.columns = ['src_id2', 'tgt_id2', 'syntactic_diff', 'plus_diff']
        print('Cache found')
    else:
        print('No cache found.')

    i = 0
    for index, row in gold_mapping.iterrows():
        if str(row['tgt_id']).lower() == str(float('NaN')) and row['label'] == 0:
            for descriptor in graph2.elements.keys():
                try:
                    src_embeddings = graph1.elements[row['src_id']].embeddings[0]
                    tgt_embeddings = graph2.elements[descriptor].embeddings[0]
                    src_columns = ["src_id"] + ["src_" + str(i) for i in range(len(src_embeddings))]
                    tgt_columns = ["tgt_id"] + ["tgt_" + str(i) for i in range(len(tgt_embeddings))]
                    sample = pd.DataFrame([[row['src_id']] + src_embeddings + [descriptor] +
                                                     tgt_embeddings + [row['label']]])
                    sample.columns = src_columns + tgt_columns + ['label']
                    # Add syntactic similarity scores.
                    if syntactic_cache_file is not None:
                        sample = append_syntactic_similarity_score(sample, syntactic_cache_file)
                    else:
                        sample = compute_append_syntactic_similarity_score(graph1, graph2, sample, 'jaccard')
                        if sample.syntactic_diff.values[0] > 0.5:
                            continue
                        if calc_PLUS_SCORE:
                            sample = compute_append_syntactic_similarity_score(graph1, graph2, sample, 'levenshtein', None, True)

                    sample = sample.drop(['src_id', 'tgt_id'], axis=1)
                    sample = sample.reset_index(drop=True)
                    sample = extend_features(sample)

                except KeyError:
                    pass
                #{'src_id': row['src_id'], 'tgt_id': descriptor, 'label': row['label']}
        if str(row['src_id']).lower() == str(float('NaN')) and row['label'] == 0:
            for descriptor in graph1.elements.keys():
                try:
                    src_embeddings = graph1.elements[descriptor].embeddings[0]
                    tgt_embeddings = graph2.elements[row['tgt_id']].embeddings[0]
                    src_columns = ["src_id"] + ["src_" + str(i) for i in range(len(src_embeddings))]
                    tgt_columns = ["tgt_id"] + ["tgt_" + str(i) for i in range(len(tgt_embeddings))]
                    sample = pd.DataFrame([[descriptor] + src_embeddings + [row['tgt_id']] +
                                           tgt_embeddings + [row['label']]])
                    sample.columns = src_columns + tgt_columns + ['label']
                    # Add syntactic similarity scores.
                    if syntactic_cache_file is not None:
                        sample = append_syntactic_similarity_score(sample, syntactic_cache_file)
                    else:
                        sample = compute_append_syntactic_similarity_score(graph1, graph2, sample, 'jaccard')
                        if sample.syntactic_diff.values[0] > 0.5:
                            continue
                        if calc_PLUS_SCORE:
                            sample = compute_append_syntactic_similarity_score(graph1, graph2, sample, 'levenshtein', None, True)

                    sample = sample.drop(['src_id', 'tgt_id'], axis=1)
                    sample = sample.reset_index(drop=True)
                    sample = extend_features(sample)

                except KeyError:
                    pass
                #{'src_id': descriptor, 'tgt_id': row['tgt_id'], 'label': row['label']}
        else:
            try:
                src_embeddings = graph1.elements[row['src_id']].embeddings[0]
                tgt_embeddings = graph2.elements[row['tgt_id']].embeddings[0]
                src_columns = ["src_id"] + ["src_" + str(i) for i in range(len(src_embeddings))]
                tgt_columns = ["tgt_id"] + ["tgt_" + str(i) for i in range(len(tgt_embeddings))]
                sample = pd.DataFrame([[row['src_id']] + src_embeddings + [row['tgt_id']] +
                                                 tgt_embeddings + [row['label']]])
                sample.columns = src_columns + tgt_columns + ['label']
                # Add syntactic similarity scores.
                if syntactic_cache_file is not None:
                    sample = append_syntactic_similarity_score(sample, syntactic_cache_file)
                else:
                    sample = compute_append_syntactic_similarity_score(graph1, graph2, sample, 'jaccard')
                    if sample.syntactic_diff.values[0] > 0.5:
                        continue
                    if calc_PLUS_SCORE:
                        sample = compute_append_syntactic_similarity_score(graph1, graph2, sample, 'levenshtein', None, True)

                sample = sample.drop(['src_id', 'tgt_id'], axis=1)
                sample = sample.reset_index(drop=True)
                sample = extend_features(sample)
            except KeyError:
                pass

        if i%1000 == 0:
            i = 1
            yield sample, None
            combined_samples = None
        else:
            if combined_samples is None:
                combined_samples = sample
            else:
                combined_samples = combined_samples.append(sample, ignore_index=True)

            #{'src_id': row['src_id'], 'tgt_id': row['tgt_id'], 'label': row['label']}

def stream_cross_product(graph1, graph2):

    i = 1
    for descriptor1, resource1 in graph1.elements.items():
        for descriptor2, resource2 in graph1.elements.items():
            src_embeddings = graph1.elements[descriptor1].embeddings[0]
            tgt_embeddings = graph2.elements[descriptor2].embeddings[0]
            src_columns = ["src_id"] + ["src_" + str(i) for i in range(len(src_embeddings))]
            tgt_columns = ["tgt_id"] + ["tgt_" + str(i) for i in range(len(tgt_embeddings))]
            sample = pd.DataFrame([[descriptor1] + src_embeddings + [descriptor2] +
                                             tgt_embeddings ])
            sample.columns = src_columns + tgt_columns
            # Add syntactic similarity scores.
            sample = compute_append_syntactic_similarity_score(graph1, graph2, sample, 'jaccard')
            sample = compute_append_syntactic_similarity_score(graph1, graph2, sample, 'levenshtein', None, True)
            sample = sample.drop(['src_id', 'tgt_id'], axis=1)
            sample = sample.reset_index(drop=True)
            sample = extend_features(sample)

            if i%1000 == 0:
                combined_samples = combined_samples.append(sample, ignore_index=True)
                i=1
                yield combined_samples
                combined_samples = None
            else:
                i=i+1
                if combined_samples is None:
                    combined_samples = sample
                else:
                    combined_samples = combined_samples.append(sample, ignore_index=True)



def prepare_data_from_file(src_path, tgt_path, gold_path, n_positive_samples=1000, n_negative_samples=1000):
    pd.options.mode.chained_assignment = None

    print("     --> Preparing data for matching ...")

    # Load the data from csv
    src = pd.read_csv(src_path, delimiter=" ", header=None, skiprows=1)
    src = src.applymap(lambda s:s.lower() if type(s) == str else s)
    tgt = pd.read_csv(tgt_path, delimiter=" ", header=None, skiprows=1)
    tgt = tgt.applymap(lambda s:s.lower() if type(s) == str else s)


    src = src.dropna(axis=1)
    tgt = tgt.dropna(axis=1)
    src.columns = ["src_id"] + ["src_" + str(i) for i in range(len(src.columns) - 1)]
    tgt.columns = ["tgt_id"] + ["tgt_" + str(i) for i in range(len(tgt.columns) - 1)]
    gold_mapping = pd.read_csv(gold_path, delimiter="\t", header=None, skiprows=1)
    gold_mapping = gold_mapping.applymap(lambda s:s.lower() if type(s) == str else s)
    gold_mapping.columns = ["gold_src_id", "gold_tgt_id", "label"]
    # Sample the data according to the parameters given.

    if gold_mapping.shape[0] < n_positive_samples:
        pn_proportion = n_negative_samples/n_positive_samples
        n_positive_samples = gold_mapping.shape[0]
        n_negative_samples = int(pn_proportion * n_positive_samples)

    gold_mapping = gold_mapping.sample(n=n_positive_samples)
    # Extract the positive samples as given in the gold_mapping-file from the source AND target dataframe.
    src.set_index("src_id")
    gold_mapping.set_index("gold_src_id")
    src_gold = pd.merge(src, gold_mapping, sort=False, how="inner", left_on="src_id", right_on="gold_src_id").drop(
        ["gold_src_id"], axis=1)
    tgt.set_index("tgt_id")
    gold_mapping = gold_mapping.reset_index(drop=True)
    gold_mapping.set_index("gold_tgt_id")
    tgt_gold = pd.merge(tgt, gold_mapping, sort=False, how="inner", left_on="tgt_id", right_on="gold_tgt_id").drop(
        ["gold_tgt_id", "gold_src_id"], axis=1)
    src_gold = src_gold.reset_index(drop=True)
    src_gold.set_index("gold_tgt_id")
    tgt_gold = tgt_gold.reset_index(drop=True)
    tgt_gold.set_index("tgt_id")
    # At this point, we have two dataframes, both only with the positive samples; now join them according
    # to the given gold_mappings, so that we have one dataframe with only positives samples containing information from
    # the source-instances and target-instances.
    positive_samples = pd.merge(src_gold, tgt_gold, left_on='gold_tgt_id', right_on='tgt_id')
    positive_samples = positive_samples.reset_index(drop=True)
    positive_samples = positive_samples.drop(["gold_tgt_id"], axis=1)

    # Create a random entity-mapping, will probably result in mostly negative samples
    src['src_random'] = np.random.randint(0, round(len(src) / n_negative_samples), size=len(src))
    src = src.reset_index(drop=True)
    src.set_index("src_random")
    tgt['tgt_random'] = np.random.randint(0, round(len(tgt) / n_negative_samples), size=len(tgt))
    tgt = tgt.reset_index(drop=True)
    tgt.set_index("tgt_random")
    negative_samples = pd.merge(src, tgt, sort=False, how="inner", left_on="src_random", right_on="tgt_random").drop(
        ["src_random", "tgt_random"], axis=1)
    negative_samples = negative_samples.reset_index(drop=True)
    # Save which IDs are contained in the set of positive samples
    positives = set(positive_samples["src_id"] + positive_samples["tgt_id"])
    # Now remove the positive samples from the negative_samples-dataframe,
    # so that we have only negative samples in there.
    negative_samples = negative_samples.loc[~(negative_samples["src_id"] + negative_samples["tgt_id"]).isin(positives)]
    # Sample the data according to the parameters given.
    negative_samples = negative_samples.sample(n=n_negative_samples)

    # Additionally, set the labels of the dataframes, i.e. 0=negative, 1=positive
    positive_samples.insert(len(positive_samples.columns), "label", 1)
    negative_samples.insert(len(negative_samples.columns), "label", 0)

    # Merge the positives with the negatives dataframe
    combined_samples = positive_samples.append(negative_samples, ignore_index=True, sort=False)
    # shuffle the rows.
    combined_samples = shuffle(combined_samples)
    combined_samples = combined_samples.reset_index(drop=True)

    combined_samples_ids = combined_samples[['src_id','tgt_id']]

    # Finally, drop the ID-columns of the positives and negatives samples dataframe,
    # as the pandas-classifier shall not be trained on them.
    combined_samples = combined_samples.drop(['src_id', 'tgt_id'], axis=1)
    positive_samples = positive_samples.drop(["src_id","tgt_id"], axis =1)
    negative_samples = negative_samples.drop(["src_id","tgt_id"], axis=1)

    return positive_samples, negative_samples, combined_samples, combined_samples_ids

def get_negatives(src, tgt, positive_samples, n_negative_samples, method='non_negative_sampling'):
    if method == 'non_negative_sampling':

        # Create a random entity-mapping, will probably result in mostly negative samples
        src['src_random'] = np.random.randint(0, round(len(src) / n_negative_samples), size=len(src))
        src = src.reset_index(drop=True)
        src.set_index("src_random")
        tgt['tgt_random'] = np.random.randint(0, round(len(tgt) / n_negative_samples), size=len(tgt))
        tgt = tgt.reset_index(drop=True)
        tgt.set_index("tgt_random")

        negative_samples = pd.merge(src, tgt, sort=False, how="inner", left_on="src_random", right_on="tgt_random").drop(
            ["src_random", "tgt_random"], axis=1)
        negative_samples = negative_samples.reset_index(drop=True)
        # Save which IDs are contained in the set of positive samples
        positives = set(positive_samples["src_id"] + positive_samples["tgt_id"])
        # Now remove the positive samples from the negative_samples-dataframe,
        # so that we have only negative samples in there.
        negative_samples = negative_samples.loc[~(negative_samples["src_id"] + negative_samples["tgt_id"]).isin(positives)]
        # Sample the data according to the parameters given.
        negative_samples = negative_samples.sample(n=n_negative_samples)

        return negative_samples

    else:
        return None




def sample_gold_data(gold_path):
    #former_gold_file = open(gold_path, "r")
    gold_mapping = pd.read_csv(gold_path, delimiter="\t", header=None, skiprows=1)
    gold_mapping = gold_mapping.applymap(lambda s: s.lower() if type(s) == str else s)
    gold_mapping.columns = ["gold_src_id", "gold_tgt_id", "label"]
    gold_mapping = gold_mapping.sample(n=round(len(gold_mapping)/2))
    try:
        os.remove(gold_path+".sampled")
    except:
        pass
    gold_mapping[['gold_src_id','gold_tgt_id']].to_csv(path_or_buf=gold_path + '.sampled', sep='\t', header=False, index=False)


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
    df['src_tgt_angle'] = cosine_similarity(a,b).diagonal()
    src_origin = np.full((len(df), src_dim), 0.0000001)
    tgt_origin = np.full((len(df), tgt_dim), 0.0000001)
    df['src_angle_to_origin'] = cosine_similarity(tgt_origin,a).diagonal()
    df['tgt_angle_to_origin'] = cosine_similarity(src_origin,b).diagonal()
    df['src_veclen'] = length(a)
    df['tgt_veclen'] = length(b)
    df['src_tgt_veclen'] = euclidean_distances(a,b).diagonal()#length(a-b)
    df.head()

    df.fillna(0, inplace = True)
    return df


def extract_non_trivial_matches(graph1, graph2, matching_ids, src_properties, tgt_properties, combined_samples,
                                positives_difference_threshold=0.7, negatives_similarity_threshold=0.75):

    non_trivial_matches = None

    for index, row in matching_ids.iterrows():
        txt1 = ""
        txt2 = ""
        elem = graph1.elements[row['src_id']]
        if src_properties is not None:
            for literal_relation in elem.literals.keys():
                if literal_relation in src_properties:
                    txt1 = txt1 + elem.literals[literal_relation]
        else:
            for literal_relation in elem.literals.keys():
                txt1 = txt1 + elem.literals[literal_relation]
        elem = graph2.elements[row['tgt_id']]
        if tgt_properties is not None:
            for literal_relation in elem.literals.keys():
                if literal_relation in tgt_properties:
                    txt2 = txt2 + elem.literals[literal_relation]
        else:
            for literal_relation in elem.literals.keys():
                txt2 = txt2 + elem.literals[literal_relation]
        txt1 = getNGrams(txt1)
        txt2 = getNGrams(txt2)
        overlap = [val for val in txt1 if val in txt2]
        # Treat positive samples different than negative samples
        # I.e. we want positive samples that are syntactically different
        # and negative samples which are syntactically similar.
        if combined_samples.loc[index, 'label'] < 0.5:
            if len(overlap) > negatives_similarity_threshold*min(len(txt1), len(txt2)):
                if non_trivial_matches is None:
                    non_trivial_matches = np.array([row])
                else:
                    non_trivial_matches = np.append(non_trivial_matches, np.array([row]), axis=0)
        else:
            if len(overlap) < positives_difference_threshold*min(len(txt1), len(txt2)):
                if non_trivial_matches is None:
                    non_trivial_matches = np.array([row])
                else:
                    non_trivial_matches = np.append(non_trivial_matches, np.array([row]), axis=0)
    non_trivial_matches = pd.DataFrame(non_trivial_matches, columns=['src_id','tgt_id'])
    return non_trivial_matches


def append_syntactic_similarity_score(df, syntactics_cache_file):
    #print("         (Cached file found. Loading ...)")
    syntactics = syntactics_cache_file

    df = pd.merge(df, syntactics, left_on=['src_id', 'tgt_id'], right_on=['src_id2', 'tgt_id2'], how='left')
    df = df.drop(['src_id2', 'tgt_id2'], axis=1)
    #print("         Complete.")
    return df


def compute_append_syntactic_similarity_score(graph1, graph2, df, method='levenshtein', properties = None, plus_differentiator=False):
    #print("         (No cached files. Starting computation for a file.)")
    global syntaxprogress
    syntaxprogress = 0
    if plus_differentiator is True:
        attname = 'plus_diff'
    else:
        attname = 'syntactic_diff'

    if method == 'levenshtein':
        print('         Levenshtein computation, progress: 0%', end="\r")
        total_size = len(df)
        ''' Here we have a problem: levenshtein distance is sensitive to the order how items are added. Thats why we
        must not iterate through the outgoing edges, but keep the order by iterating over all possible properties
        '''
        for index, row in df.iterrows():
            syntaxprogress=syntaxprogress+1
            if syntaxprogress%20==0:
                print('         Levenshtein computation, progress: '+str(int((100*syntaxprogress/(total_size))))+'%', end="\r")
            txt1 = ""
            for property in [p for p in graph1.literal_properties if p in graph1.elements[row['src_id']].literals.keys()]:
                if properties is not None:# and plus_differentiator is False:
                    if not property in properties:
                        continue
                txt1 = txt1 + graph1.elements[row['src_id']].literals[property]
            txt2 = ""
            for property in [p for p in graph2.literal_properties if p in graph2.elements[row['tgt_id']].literals.keys()]:
                if properties is not None:# and plus_differentiator is False:
                    if not property in properties:
                        continue
                txt2 = txt2 + graph2.elements[row['tgt_id']].literals[property]
            if len(ngrams1) == 0 and len(ngram2) == 0:
                 df.loc[index, attname] = 1.0
            else:
                 df.loc[index, attname] = 1.0 - float(ngrams_intersection) / float(max(len(ngrams1),len(ngrams2)))
        #print("         Complete.")
        print('         Levenshtein computation, progress: 100%')
        return df

    elif method == 'jaccard':
        print('         Jaccard computation, progress: 0%', end="\r")
        total_size = len(df)
        for index, row in df.iterrows():
            syntaxprogress=syntaxprogress+1
            if syntaxprogress%20==0:
                print('         Jaccard computation, progress: '+str(int((100*syntaxprogress/(total_size))))+'%', end="\r")
            txt1 = ""
            for property in [p for p in graph1.literal_properties if
                             p in graph1.elements[row['src_id']].literals.keys()]:
                if properties is not None:# and plus_differentiator is False:
                    if not property in properties:
                        continue
                txt1 = txt1 + graph1.elements[row['src_id']].literals[property]
            txt2 = ""
            for property in [p for p in graph2.literal_properties if
                             p in graph2.elements[row['tgt_id']].literals.keys()]:
                if properties is not None:# and plus_differentiator is False:
                    if not property in properties:
                        continue
                txt2 = txt2 + graph2.elements[row['tgt_id']].literals[property]
            ngrams1 = getNGrams(txt1)
            ngrams2 = getNGrams(txt2)
            ngrams_intersection = len([value for value in ngrams1 if value in ngrams2])
            val = 1.0 - float(ngrams_intersection) / float(max(len(ngrams1),len(ngrams2)))
            df.loc[index, attname] = 1.0 - float(ngrams_intersection) / float(max(len(ngrams1),len(ngrams2)))
        print('         Jaccard computation, progress: 100%')
        #print("         Complete.")
        return df


def aggregate_to_dict(indices):
    tmp_tgt_ind = dict()
    for index in indices:
            if index in tmp_tgt_ind.keys():
                tmp_tgt_ind[index] = tmp_tgt_ind[index] + 1
            else:
                tmp_tgt_ind[index] = 1
    return tmp_tgt_ind

def iterative_levenshtein(s, t):
        s = str(s)
        t = str(t)
        if len(s) == 0 or len(t) == 0:
            return 999999

        """
            iterative_levenshtein(s, t) -> ldist
            ldist is the Levenshtein distance between the strings
            s and t.
            For all i and j, dist[i,j] will contain the Levenshtein
            distance between the first i characters of s and the
            first j characters of t
        """
        rows = len(s)+1
        cols = len(t)+1
        dist = [[0 for x in range(cols)] for x in range(rows)]
        # source prefixes can be transformed into empty strings
        # by deletions:
        for i in range(1, rows):
            dist[i][0] = i
        # target prefixes can be created from an empty source string
        # by inserting the characters
        for i in range(1, cols):
            dist[0][i] = i

        for col in range(1, cols):
            for row in range(1, rows):
                if s[row-1] == t[col-1]:
                    cost = 0
                else:
                    cost = 1
                dist[row][col] = min(dist[row-1][col] + 1,      # deletion
                                     dist[row][col-1] + 1,      # insertion
                                     dist[row-1][col-1] + cost) # substitution
        return dist[row][col]

def levenshtein_distance(word1, word2):
   ''' Returns an integer representing the Levenshtein distance between two words '''

   table = np.zeros((len(word1) + 1, len(word2) + 1), dtype=int)
   for line in range(len(word1) + 1):
       table[line][0] = line
   for column in range(len(word2) + 1):
       table[0][column] = column

   for line in range(1, len(word1) + 1):
       for column in range(1, len(word2) + 1):
           if word1[line - 1] == word2[column - 1]:
               substitution_cost = 0
           else:
               substitution_cost = 1

           table[line, column] = min(table[line - 1][column] + 1,
                                     table[line][column - 1] + 1,
                                     table[line - 1][column - 1] + substitution_cost)

   return table[len(word1)][len(word2)]

