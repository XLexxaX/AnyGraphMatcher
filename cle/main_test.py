import os
from cle.configurations.Configuration import Configuration, InternalGoldStandard
from cle.configurations.ConfigurationHandler import ConfigurationHandler
from cle.configurations.PipelineTools import Pipeline, PipelineDataTuple
from cle.loadkg.loadWithRdflib import load_kg_with_rdflib_ttl_interface
from cle.graphdatatools import GraphToolbox
from cle.wordembedding import concat_combiner, W2V_1InterfaceWrapper, SimpleTriplesEmbedder_1, PseudoD2V_1InterfaceWrapper_2
from cle.visualization import CategoriesVisualizer, TypeVisualizer, StratifiedVisualizer, FullVisualizer, TSNEInterface, EmbeddingSaver
from cle.matcher import EmbeddingMatcher, Matcher, OAEIMatcher, FlatMatcher
from cle.sentencegenerator import ReadSentencesInterfaceWrapper

#from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

package_directory = os.path.dirname(os.path.abspath(__file__))

def main():



    src_triples = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_full_strings',
                               'graph_triples_hilti_erp.nt')
    tgt_triples = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_full_strings',
                               'graph_triples_hilti_web.nt')
    src_corpus = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_full_strings',
                              'corpus_hilti_erp.txt')
    tgt_corpus = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_full_strings',
                              'corpus_hilti_web.txt')
    gold_mapping = InternalGoldStandard({'trainsets':
                                            [os.path.join(package_directory, '..', 'data',
                                            'sap_hilti_data', 'sap_hilti_full_strings', 'train_simple_sap_hilti.csv'),
                                            os.path.join(package_directory, '..', 'data', 'sap_hilti_data',
                                            'sap_hilti_full_strings', 'train_hard_sap_hilti.csv')],
                                         'testsets': [os.path.join(package_directory, '..', 'data',
                                            'sap_hilti_data', 'sap_hilti_full_strings', 'test_simple_sap_hilti.csv'),
                                            os.path.join(package_directory, '..', 'data', 'sap_hilti_data',
                                            'sap_hilti_full_strings', 'test_hard_sap_hilti.csv')]
                                        })
    dim = 20
    model = LogisticRegression()
    src_properties = None#["http://rdata2graph.sap.com/hilti_erp/property/mara_fert.maktx"]
    tgt_properties = None#["http://rdata2graph.sap.com/hilti_web/property/products.name"]




    ##name = "W2V_1 muse xgb with 50k only on embeddings"
    ##pipeline = Pipeline()
    ##line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    ##line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    ##line_a = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_a),
    ##                              PipelineDataTuple(src_corpus))
    ##line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    ##line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    ##line_b = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_b),
    ##                              PipelineDataTuple(tgt_corpus))
    ##line_ab = pipeline.append_step(W2V_1InterfaceWrapper.interface, PipelineDataTuple(line_a, line_b), PipelineDataTuple(dim))
    ##line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_ab), None)
    ##line_ab = pipeline.append_step(muse.interface, PipelineDataTuple(line_ab), PipelineDataTuple(gold_mapping))
    ##line_ab = pipeline.append_step(EmbeddingMatcher.interface, PipelineDataTuple(line_ab),
    ##                               PipelineDataTuple(model))
    ##line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
    ##line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    ##line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    ##line_ab = pipeline.append_step(FullVisualizer.interface, PipelineDataTuple(line_ab), None)
##
    ##configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
    ##                              pipeline, src_properties, tgt_properties)
    ##configuration_handler = ConfigurationHandler()
    ##configuration_handler.execute(configuration)
##
##
##
##
    ##name = "W2V_1 muse xgb with 50k on embeddings and sim"
    ##pipeline = Pipeline()
    ##line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    ##line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    ##line_a = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_a),
    ##                              PipelineDataTuple(src_corpus))
    ##line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    ##line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    ##line_b = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_b),
    ##                              PipelineDataTuple(tgt_corpus))
    ##line_ab = pipeline.append_step(W2V_1InterfaceWrapper.interface, PipelineDataTuple(line_a, line_b), PipelineDataTuple(dim))
    ##line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_ab), None)
    ##line_ab = pipeline.append_step(muse.interface, PipelineDataTuple(line_ab), PipelineDataTuple(gold_mapping))
    ##line_ab = pipeline.append_step(EmbeddingMatcher.interface, PipelineDataTuple(line_ab),
    ##                               PipelineDataTuple(model))
    ##line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
    ##line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    ##line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    ##line_ab = pipeline.append_step(FullVisualizer.interface, PipelineDataTuple(line_ab), None)
##
    ##configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
    ##                              pipeline, src_properties, tgt_properties)
    ##configuration_handler = ConfigurationHandler()
    ##configuration_handler.execute(configuration)

    name = "jacc_no_schema_given"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    #line_a = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_a),
    #                              PipelineDataTuple(src_corpus))
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    #line_b = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_b),
    #                              PipelineDataTuple(tgt_corpus))
    line_ab = pipeline.append_step(PseudoD2V_1InterfaceWrapper_2.interface, PipelineDataTuple(line_a, line_b), PipelineDataTuple(dim))
    line_ab = pipeline.append_step(SimpleTriplesEmbedder_1.interface, PipelineDataTuple(line_a, line_b), PipelineDataTuple(dim))

    line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(muse.interface, PipelineDataTuple(line_ab), PipelineDataTuple(gold_mapping))
    line_ab = pipeline.append_step(FlatMatcher.interface, PipelineDataTuple(line_ab),
                                   PipelineDataTuple(model))
    #line_ab = pipeline.append_step(TSNEInterface.interface, PipelineDataTuple(line_ab), PipelineDataTuple(2))
    #line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(FullVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(EmbeddingSaver.interface, PipelineDataTuple(line_ab), None)

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
                                  pipeline, src_properties, tgt_properties)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)


    #name = "test2"
    #pipeline = Pipeline()
    #line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    #line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    #line_a = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_a),
    #                              PipelineDataTuple(src_corpus))
    #line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    #line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    #line_b = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_b),
    #                              PipelineDataTuple(tgt_corpus))
    #line_ab = pipeline.append_step(W2V_1InterfaceWrapper.interface, PipelineDataTuple(line_a, line_b), PipelineDataTuple(dim))
    #line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_ab), None)
    ##line_ab = pipeline.append_step(muse.interface, PipelineDataTuple(line_ab), PipelineDataTuple(gold_mapping))
    #line_ab = pipeline.append_step(EmbeddingMatcher.interface, PipelineDataTuple(line_ab),
    #                               PipelineDataTuple(model))
    #line_ab = pipeline.append_step(EmbeddingSaver.interface, PipelineDataTuple(line_ab), None)

    #configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
    #                              pipeline, src_properties, tgt_properties)
    #configuration_handler = ConfigurationHandler()
    #configuration_handler.execute(configuration)
#
#
   #name = "BBBBBBBB"
   #pipeline = Pipeline()
   #line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
   #line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
   #line_a = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_a),
   #                              PipelineDataTuple(src_corpus))
   #line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
   #line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
   #line_b = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_b),
   #                              PipelineDataTuple(tgt_corpus))
   #line_ab = pipeline.append_step(SimpleTriplesEmbedder_1.interface, PipelineDataTuple(line_a, line_b), PipelineDataTuple(dim))
   #line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_ab), None)
   #line_ab = pipeline.append_step(EmbeddingMatcher.interface, PipelineDataTuple(line_ab),
   #                               PipelineDataTuple(model))
   #line_ab = pipeline.append_step(EmbeddingSaver.interface, PipelineDataTuple(line_ab), None)
#
   #configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
   #                              pipeline, src_properties, tgt_properties)
   #configuration_handler = ConfigurationHandler()
   #configuration_handler.execute(configuration)

if __name__ == '__main__':
    main()
