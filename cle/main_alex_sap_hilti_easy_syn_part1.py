

import os
from cle.configurations.Configuration import Configuration, InternalGoldStandard
from cle.configurations.ConfigurationHandler import ConfigurationHandler
from cle.configurations.PipelineTools import Pipeline, PipelineDataTuple
from cle.loadkg.loadWithRdflib import load_kg_with_rdflib_ttl_interface
from cle.graphdatatools import GraphToolbox
from cle.wordembedding import W2VInterfaceWrapper, D2VInterfaceWrapper, PseudoD2VInterfaceWrapper, \
    W2V_1InterfaceWrapper, D2V_1InterfaceWrapper, PseudoD2V_1InterfaceWrapper, SimpleTriplesEmbedder, \
    SimpleTriplesEmbedder_1, concat_combiner
from cle.visualization import CategoriesVisualizer, StratifiedVisualizer, TypeVisualizer, FullVisualizer, TSNEInterface, \
    EmbeddingSaver
from cle.sentencegenerator import ReadSentencesInterfaceWrapper
from cle.matcher import Matcher
from xgboost import XGBClassifier

package_directory = os.path.dirname(os.path.abspath(__file__))

def main():
    main_ngram_string()

def main_ngram_string():
    logfile = os.path.join(package_directory, '..', 'results.log')

    src_triples = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_3grams',
                               'graph_triples_hilti_erp.nt')
    tgt_triples = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_3grams',
                               'graph_triples_hilti_web.nt')
    src_corpus = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_3grams',
                              'corpus_hilti_erp.txt')
    tgt_corpus = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_3grams',
                              'corpus_hilti_web.txt')
    gold_mapping = InternalGoldStandard({'trainsets':
                                            [os.path.join(package_directory, '..', 'data',
                                            'sap_hilti_data', 'sap_hilti_3grams', 'train_simple_sap_hilti.csv'),
                                            os.path.join(package_directory, '..', 'data', 'sap_hilti_data',
                                            'sap_hilti_3grams', 'train_hard_sap_hilti.csv')],
                                         'testsets': [os.path.join(package_directory, '..', 'data',
                                            'sap_hilti_data', 'sap_hilti_3grams', 'test_simple_sap_hilti.csv'),
                                            os.path.join(package_directory, '..', 'data', 'sap_hilti_data',
                                            'sap_hilti_3grams', 'test_hard_sap_hilti.csv')]
                                        })
    dim = 20
    model = XGBClassifier()
    src_properties = ["http://rdata2graph.sap.com/hilti_erp/property/mara_fert.maktx"]
    tgt_properties = ["http://rdata2graph.sap.com/hilti_web/property/products.name"]




    name = "3gram: simpletriplesembedding xgb"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(SimpleTriplesEmbedder.interface, PipelineDataTuple(line_a), PipelineDataTuple(dim))
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(SimpleTriplesEmbedder.interface, PipelineDataTuple(line_b), PipelineDataTuple(dim))
    line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_a, line_b), None)
    line_ab = pipeline.append_step(Matcher.interface, PipelineDataTuple(line_ab),
                                   PipelineDataTuple(model))
    line_ab = pipeline.append_step(TSNEInterface.interface, PipelineDataTuple(line_ab), PipelineDataTuple(2))
    line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(FullVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(EmbeddingSaver.interface, PipelineDataTuple(line_ab), None)
    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
                                  pipeline, src_properties, tgt_properties)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)



    name = "3gram: simpletriplesembedding_1 xgb"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    line_ab = pipeline.append_step(SimpleTriplesEmbedder_1.interface, PipelineDataTuple(line_a, line_b), PipelineDataTuple(dim))
    line_ab = pipeline.append_step(Matcher.interface, PipelineDataTuple(line_ab),
                                   PipelineDataTuple(model))
    line_ab = pipeline.append_step(TSNEInterface.interface, PipelineDataTuple(line_ab), PipelineDataTuple(2))
    line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(FullVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(EmbeddingSaver.interface, PipelineDataTuple(line_ab), None)
    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
                                  pipeline, src_properties, tgt_properties)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)


    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
                                  pipeline, src_properties, tgt_properties)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)





    name = "3gram: w2v xgb"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_a),
                                  PipelineDataTuple(src_corpus))
    line_a = pipeline.append_step(W2VInterfaceWrapper.interface, PipelineDataTuple(line_a), PipelineDataTuple(dim))
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_b),
                                  PipelineDataTuple(tgt_corpus))
    line_b = pipeline.append_step(W2VInterfaceWrapper.interface, PipelineDataTuple(line_b), PipelineDataTuple(dim))
    line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_a, line_b), None)
    line_ab = pipeline.append_step(Matcher.interface, PipelineDataTuple(line_ab),
                                   PipelineDataTuple(model))
    line_ab = pipeline.append_step(TSNEInterface.interface, PipelineDataTuple(line_ab), PipelineDataTuple(2))
    line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(FullVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(EmbeddingSaver.interface, PipelineDataTuple(line_ab), None)

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
                                  pipeline, src_properties, tgt_properties)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)






    name = "3gram: d2v xgb"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(D2VInterfaceWrapper.interface, PipelineDataTuple(line_a), PipelineDataTuple(dim))
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(D2VInterfaceWrapper.interface, PipelineDataTuple(line_b), PipelineDataTuple(dim))
    line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_a, line_b), None)
    line_ab = pipeline.append_step(Matcher.interface, PipelineDataTuple(line_ab),
                                   PipelineDataTuple(model))
    line_ab = pipeline.append_step(TSNEInterface.interface, PipelineDataTuple(line_ab), PipelineDataTuple(2))
    line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(FullVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(EmbeddingSaver.interface, PipelineDataTuple(line_ab), None)

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
                                  pipeline, src_properties, tgt_properties)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)




    name = "3gram: pseudod2v xgb"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(PseudoD2VInterfaceWrapper.interface, PipelineDataTuple(line_a), PipelineDataTuple(dim))
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(PseudoD2VInterfaceWrapper.interface, PipelineDataTuple(line_b), PipelineDataTuple(dim))
    line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_a, line_b), None)
    line_ab = pipeline.append_step(Matcher.interface, PipelineDataTuple(line_ab),
                                   PipelineDataTuple(model))
    line_ab = pipeline.append_step(TSNEInterface.interface, PipelineDataTuple(line_ab), PipelineDataTuple(2))
    line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(FullVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(EmbeddingSaver.interface, PipelineDataTuple(line_ab), None)

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
                                  pipeline, src_properties, tgt_properties)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)







    name = "3gram: W2V_1 xgb"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_a),
                                  PipelineDataTuple(src_corpus))
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_b),
                                  PipelineDataTuple(tgt_corpus))
    line_ab = pipeline.append_step(W2V_1InterfaceWrapper.interface, PipelineDataTuple(line_a, line_b), PipelineDataTuple(dim))
    line_ab = pipeline.append_step(Matcher.interface, PipelineDataTuple(line_ab),
                                   PipelineDataTuple(model))
    line_ab = pipeline.append_step(TSNEInterface.interface, PipelineDataTuple(line_ab), PipelineDataTuple(2))
    line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(FullVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(EmbeddingSaver.interface, PipelineDataTuple(line_ab), None)

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
                                  pipeline, src_properties, tgt_properties)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)






    name = "3gram: D2V_1 xgb"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    line_ab = pipeline.append_step(D2V_1InterfaceWrapper.interface, PipelineDataTuple(line_a, line_b), PipelineDataTuple(dim))
    line_ab = pipeline.append_step(Matcher.interface, PipelineDataTuple(line_ab),
                                   PipelineDataTuple(model))
    line_ab = pipeline.append_step(TSNEInterface.interface, PipelineDataTuple(line_ab), PipelineDataTuple(2))
    line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(FullVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(EmbeddingSaver.interface, PipelineDataTuple(line_ab), None)

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
                                  pipeline, src_properties, tgt_properties)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)



    name = "3gram: pseudoD2V_1 xgb"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    line_ab = pipeline.append_step(PseudoD2V_1InterfaceWrapper.interface, PipelineDataTuple(line_a, line_b), PipelineDataTuple(dim))
    line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(Matcher.interface, PipelineDataTuple(line_ab),
                                   PipelineDataTuple(model))
    line_ab = pipeline.append_step(TSNEInterface.interface, PipelineDataTuple(line_ab), PipelineDataTuple(2))
    line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(FullVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(EmbeddingSaver.interface, PipelineDataTuple(line_ab), None)

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
                                  pipeline, src_properties, tgt_properties)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)



if __name__ == '__main__':
    main_ngram_string()
