
import os
import sys

from StringMatching import StringMatcher_Interface
from configurations.Configuration import Configuration
from configurations.InternalGoldStandard import InternalGoldStandard
from configurations.InternalProperties import InternalProperties
from configurations.ConfigurationHandler import ConfigurationHandler
from configurations.PipelineTools import Pipeline, PipelineDataTuple
from loadkg.loadWithRdflib import load_kg_with_rdflib_ttl_interface
from graphdatatools import GraphToolbox
from wordembedding import W2VInterfaceWrapper, D2VInterfaceWrapper, PseudoD2VInterfaceWrapper, \
    W2V_1InterfaceWrapper, D2V_1InterfaceWrapper, PseudoD2V_1InterfaceWrapper, SimpleTriplesEmbedder, \
    SimpleTriplesEmbedder_1, concat_combiner, ResourceRelationsEmbeddingWrapper, SimpleLiteralsEmbedder_1, \
    WalkEmbedder_1, WalkD2V_1Embedder
from visualization import CategoriesVisualizer, StratifiedVisualizer, TypeVisualizer, FullVisualizer, \
    EmbeddingSaver, TSNEInterface
from sentencegenerator import ReadSentencesInterfaceWrapper
from matcher import UnsupervisedRankMatcher, SupervisedRankMatcher
#from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from configurations.DiskDataPreparation import prepare_dir, copytree

import shutil
import numpy
import scipy
import gensim
import sklearn
import pandas
import nltk
import rdflib
import editdistance
import xopen
import xgboost
import urllib3
import joblib
import matplotlib

package_directory = os.path.dirname(os.path.abspath(__file__))

def main(source, target, possible_matches, trainset):


    src_corpus = None
    tgt_corpus = None
    src_triples = source
    tgt_triples = target
    if trainset:
        trainset = [trainset]
    else:
        trainset = []
    gold_mapping = InternalGoldStandard({'trainsets': trainset,
                                         'testsets': [possible_matches]
                                        })
    dim = 100
    model = LogisticRegression()#XGBClassifier()
    properties = InternalProperties({'src_labels': ["http://www.w3.org/2000/01/rdf-schema#label"],
                                     'tgt_labels': ["http://www.w3.org/2000/01/rdf-schema#label"],
                                     'category': "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                                     'class': "http://www.w3.org/2002/07/owl#class",
                                     'property': "http://www.w3.org/1999/02/22-rdf-syntax-ns#property"})
    #properties = InternalProperties({'src_labels': ["http://rdata2graph.sap.com/hilti_erp/property/mara_fert.maktx"],
    #                                 'tgt_labels': ["http://rdata2graph.sap.com/hilti_web/property/products.name"],
    #                                 'category': "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    #                                 'class': "http://www.w3.org/2002/07/owl#class",
    #                                 'property': "http://www.w3.org/1999/02/22-rdf-syntax-ns#property"})
    use_streams = False


    name = "cli_task"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    #line_a = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_a),
    #                              PipelineDataTuple(src_corpus))
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    #line_b = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_b),
    #                              PipelineDataTuple(tgt_corpus))
    line_ab = pipeline.append_step(WalkEmbedder_1.interface, PipelineDataTuple(line_a, line_b),
                                   PipelineDataTuple(dim, 'steps', False, 1))
    line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_ab), None)

    if len(trainset)>0:
        line_ab = pipeline.append_step(SupervisedRankMatcher.interface, PipelineDataTuple(line_ab), None)
    else:
        line_ab = pipeline.append_step(UnsupervisedRankMatcher.interface, PipelineDataTuple(line_ab), None)

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
                                  pipeline, properties, use_streams, False, True)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)

    cli_resultdir = os.path.join(configuration.rundir, "..", "cli_result")
    try:
        shutil.rmtree(cli_resultdir)
    except FileNotFoundError:
        pass
    prepare_dir(cli_resultdir)
    copytree(configuration.rundir, cli_resultdir)


    #copy

if __name__ == '__main__':
    #python agm_cli.py -s "../data/oaei_data/graph_triples_darkscape.nt" -t "../data/oaei_data/graph_triples_oldschoolrunescape.nt" -p "../data/oaei_data/possible_matches.csv" -g "../data/oaei_data/trainset.csv"
    #C:\ProgramData\Anaconda3\envs\py36\Python
    #python agm_cli.py -s "C:\Users\allue\oaei_track_cache\tmpdata\graph_triples_source.nt" -t "C:\Users\allue\oaei_track_cache\tmpdata\graph_triples_target.nt" -p "C:\Users\allue\oaei_track_cache\tmpdata\possible_matches.csv" -g "C:\Users\allue\oaei_track_cache\tmpdata\artificial_gold_standard.csv"
    #main("../data/oaei_data/graph_triples_darkscape.nt", "../data/oaei_data/graph_triples_oldschoolrunescape.nt", "../data/oaei_data/possible_matches.csv", None)# "../data/oaei_data/trainset.csv"

        from optparse import OptionParser, OptionGroup

        optparser = OptionParser(
            description="An integrated, fully automatic, semantic identity resolution system")
        optparser.add_option("-s", "--source", default=None, dest="source", help="Path to the source .nt-file")
        optparser.add_option("-t", "--target", default=None, dest="target", help="Path to the target .nt-file")
        optparser.add_option("-p", "--possible_matches", default=None, dest="possible_matches", help="Path to the .csv-file with possible matches")
        optparser.add_option("-g", "--gold_standard", default=None, dest="gold_standard", help="Path to the .csv-file with gold standard entries")
        (options, args) = optparser.parse_args()
        assert options.source, "A source data set must be provided"
        assert options.target, "A target data set must be provided"
        assert options.possible_matches, "A data set with possible matches must be provided"
        main(options.source, options.target, options.possible_matches, options.gold_standard)
