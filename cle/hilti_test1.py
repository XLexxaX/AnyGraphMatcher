
import os
from configurations.Configuration import Configuration, InternalGoldStandard
from configurations.ConfigurationHandler import ConfigurationHandler
from configurations.PipelineTools import Pipeline, PipelineDataTuple
from loadkg.loadWithRdflib import load_kg_with_rdflib_ttl_interface
from graphdatatools import GraphToolbox
from wordembedding import W2VInterfaceWrapper, D2VInterfaceWrapper, PseudoD2VInterfaceWrapper, \
    W2V_1InterfaceWrapper, D2V_1InterfaceWrapper, PseudoD2V_1InterfaceWrapper, SimpleTriplesEmbedder, \
    SimpleTriplesEmbedder_1, concat_combiner, ResourceRelationsEmbeddingWrapper, muse, SimpleLiteralsEmbedder_1, \
    WalkEmbedder_1, WalkD2V_1Embedder, RelationsWalkEmbedder_1
from visualization import CategoriesVisualizer, StratifiedVisualizer, TypeVisualizer, FullVisualizer, \
    EmbeddingSaver, TSNEInterface
from sentencegenerator import ReadSentencesInterfaceWrapper
from matcher import FlatMatcher, StableRankMatcher, PredictionToXMLConverter
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from StringMatching import StringMatcher_Interface

package_directory = os.path.dirname(os.path.abspath(__file__))

def main():



    logfile = os.path.join(package_directory, '..', 'results.log')

    src_triples = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'balanced_walks',
                               'graph_triples_hilti_erp.nt')
    tgt_triples = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'balanced_walks',
                               'graph_triples_hilti_web.nt')
    src_corpus = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'balanced_walks',
                              'corpus_hilti_erp.txt')
    tgt_corpus = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'balanced_walks',
                              'corpus_hilti_web.txt')
    gold_mapping = InternalGoldStandard({'trainsets':
                                            [os.path.join(package_directory, '..', 'data',
                                            'sap_hilti_data', 'balanced_walks', 'final_trainset.csv')],
                                         'testsets': [os.path.join(package_directory, '..', 'data',
                                            'sap_hilti_data', 'balanced_walks', 'possible_matches.csv')]
                                        })
    dim = 20
    model = XGBClassifier()
    src_properties = ["http://rdata2graph.sap.com/hilti_erp/property/mara_fert.maktx"]
    tgt_properties = ["http://rdata2graph.sap.com/hilti_web/property/products.name"]



    name = "OAEI_d2v_batch_walklength1"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    line_ab = pipeline.append_step(WalkD2V_1Embedder.interface, PipelineDataTuple(line_a, line_b),
                                   PipelineDataTuple(dim, 'batch', False, 1))
    line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(muse.interface, PipelineDataTuple(line_ab), PipelineDataTuple(gold_mapping))
    line_ab = pipeline.append_step(FlatMatcher.interface, PipelineDataTuple(line_ab),
                                   PipelineDataTuple(model))
    #line_ab = pipeline.append_step(TSNEInterface.interface, PipelineDataTuple(line_ab), PipelineDataTuple(2))
    line_ab = pipeline.append_step(EmbeddingSaver.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(StableRankMatcher.interface, PipelineDataTuple(line_ab), None)
    

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
                                  pipeline, src_properties, tgt_properties, calc_PLUS_SCORE=False, use_cache=False, use_streams=False)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)



    #name = "OAEI_d2v_batch_walklength3"
    #pipeline = Pipeline()
    #line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    #line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    #line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    #line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    #line_ab = pipeline.append_step(WalkD2V_1Embedder.interface, PipelineDataTuple(line_a, line_b),
    #                               PipelineDataTuple(dim, 'batch', False, 3))
    #line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_ab), None)
    ##line_ab = pipeline.append_step(muse.interface, PipelineDataTuple(line_ab), PipelineDataTuple(gold_mapping))
    #line_ab = pipeline.append_step(FlatMatcher.interface, PipelineDataTuple(line_ab),
    #                               PipelineDataTuple(model))
    ##line_ab = pipeline.append_step(TSNEInterface.interface, PipelineDataTuple(line_ab), PipelineDataTuple(2))
    #line_ab = pipeline.append_step(EmbeddingSaver.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(StableRankMatcher.interface, PipelineDataTuple(line_ab), None)
    #
#
    #configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
    #                              pipeline, src_properties, tgt_properties, calc_PLUS_SCORE=False, use_cache=False, use_streams=False)
    #configuration_handler = ConfigurationHandler()
    #configuration_handler.execute(configuration)


    name = "OAEI_relw2v_steps_walklength1"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    line_ab = pipeline.append_step(RelationsWalkEmbedder_1.interface, PipelineDataTuple(line_a, line_b),
                                   PipelineDataTuple(dim, 'steps', False, 1))
    line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(muse.interface, PipelineDataTuple(line_ab), PipelineDataTuple(gold_mapping))
    line_ab = pipeline.append_step(FlatMatcher.interface, PipelineDataTuple(line_ab),
                                   PipelineDataTuple(model))
    #line_ab = pipeline.append_step(TSNEInterface.interface, PipelineDataTuple(line_ab), PipelineDataTuple(2))
    line_ab = pipeline.append_step(EmbeddingSaver.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(StableRankMatcher.interface, PipelineDataTuple(line_ab), None)
    

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
                                  pipeline, src_properties, tgt_properties, calc_PLUS_SCORE=False, use_cache=False, use_streams=False)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)


    #name = "OAEI_relw2v_batch_walklength1"
    #pipeline = Pipeline()
    #line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    #line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    #line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    #line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    #line_ab = pipeline.append_step(RelationsWalkEmbedder_1.interface, PipelineDataTuple(line_a, line_b),
    #                               PipelineDataTuple(dim, 'batch', False, 1))
    #line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_ab), None)
    ##line_ab = pipeline.append_step(muse.interface, PipelineDataTuple(line_ab), PipelineDataTuple(gold_mapping))
    #line_ab = pipeline.append_step(FlatMatcher.interface, PipelineDataTuple(line_ab),
    #                               PipelineDataTuple(model))
    ##line_ab = pipeline.append_step(TSNEInterface.interface, PipelineDataTuple(line_ab), PipelineDataTuple(2))
    #line_ab = pipeline.append_step(EmbeddingSaver.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(StableRankMatcher.interface, PipelineDataTuple(line_ab), None)
    #
#
    #configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
    #                              pipeline, src_properties, tgt_properties, calc_PLUS_SCORE=False, use_cache=False, use_streams=False)
    #configuration_handler = ConfigurationHandler()
    #configuration_handler.execute(configuration)
#
#
    #name = "OAEI_relw2v_steps_walklength3"
    #pipeline = Pipeline()
    #line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    #line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    #line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    #line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    #line_ab = pipeline.append_step(RelationsWalkEmbedder_1.interface, PipelineDataTuple(line_a, line_b),
    #                               PipelineDataTuple(dim, 'steps', False, 3))
    #line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_ab), None)
    ##line_ab = pipeline.append_step(muse.interface, PipelineDataTuple(line_ab), PipelineDataTuple(gold_mapping))
    #line_ab = pipeline.append_step(FlatMatcher.interface, PipelineDataTuple(line_ab),
    #                               PipelineDataTuple(model))
    ##line_ab = pipeline.append_step(TSNEInterface.interface, PipelineDataTuple(line_ab), PipelineDataTuple(2))
    #line_ab = pipeline.append_step(EmbeddingSaver.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(StableRankMatcher.interface, PipelineDataTuple(line_ab), None)
    #
#
    #configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
    #                              pipeline, src_properties, tgt_properties, calc_PLUS_SCORE=False, use_cache=False, use_streams=False)
    #configuration_handler = ConfigurationHandler()
    #configuration_handler.execute(configuration)


if __name__ == '__main__':
    main()
