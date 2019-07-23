import os
import os
from configurations.Configuration import Configuration
from configurations.ConfigurationHandler import ConfigurationHandler
from configurations.PipelineTools import Pipeline, PipelineDataTuple
from loadkg.loadWithRdflib import load_kg_with_rdflib_ttl_interface
from graphdatatools import GraphToolbox
from wordembedding import W2VInterfaceWrapper, D2VInterfaceWrapper, PseudoD2VInterfaceWrapper, \
    W2V_1InterfaceWrapper, D2V_1InterfaceWrapper, PseudoD2V_1InterfaceWrapper, SimpleTriplesEmbedder, \
    SimpleTriplesEmbedder_1, concat_combiner, ResourceRelationsEmbeddingWrapper, muse
from visualization import CategoriesVisualizer, StratifiedVisualizer, TypeVisualizer, FullVisualizer
from sentencegenerator import ReadSentencesInterfaceWrapper
from matcher import Matcher, EmbeddingMatcher
from xgboost import XGBClassifier

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
    gold_mapping = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_full_strings',
                                'train_simple_sap_hilti.csv')

    dim = 3
    #model = make_pipeline(PolynomialFeatures(6), Ridge())#DecisionTreeClassifier() #make_pipeline(PolynomialFeatures(8), Ridge())
    #model = sklearn.linear_model.LinearRegression()
    #from sklearn.ensemble import RandomForestRegressor
    #model = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
    #model = LinearSVC(C=0.01, class_weight=None, dual=True, fit_intercept=True,
    #                  intercept_scaling=1, loss='squared_hinge', max_iter=1000,
    #                  multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)
    from sklearn.linear_model import LogisticRegression
    model = XGBClassifier()
    src_properties = ["http://rdata2graph.sap.com/hilti_erp/property/mara_fert.maktx"]
    tgt_properties = ["http://rdata2graph.sap.com/hilti_web/property/products.name"]


    name = "jaccard_no_props_given"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    line_ab = pipeline.append_step(W2V_1InterfaceWrapper.interface, PipelineDataTuple(line_a, line_b), PipelineDataTuple(dim))
    line_ab = pipeline.append_step(muse.interface, PipelineDataTuple(line_a, line_b), PipelineDataTuple(gold_mapping))
    line_ab = pipeline.append_step(EmbeddingMatcher.interface, PipelineDataTuple(line_ab),
                                   PipelineDataTuple(model))
    line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_a, line_b), None)
    line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(FullVisualizer.interface, PipelineDataTuple(line_ab), None)
    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
                                  pipeline, src_properties, tgt_properties)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)



    #name = "w2v xgb"
    #pipeline = Pipeline()
    #line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    #line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    #line_a = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_a),
    #                              PipelineDataTuple(src_corpus))
    #line_a = pipeline.append_step(W2VInterfaceWrapper.interface, PipelineDataTuple(line_a), PipelineDataTuple(dim))
    #line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    #line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    #line_b = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_b),
    #                              PipelineDataTuple(tgt_corpus))
    #line_b = pipeline.append_step(W2VInterfaceWrapper.interface, PipelineDataTuple(line_b), PipelineDataTuple(dim))
    #line_ab = pipeline.append_step(Matcher.interface, PipelineDataTuple(line_a, line_b),
    #                               PipelineDataTuple(model))
    #line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
#
    #configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
    #                              pipeline, src_properties, tgt_properties)
    #configuration_handler = ConfigurationHandler()
    #configuration_handler.execute(configuration)
#
#
#
#
#
    #name = "d2v xgb"
    #pipeline = Pipeline()
    #line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    #line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    #line_a = pipeline.append_step(D2VInterfaceWrapper.interface, PipelineDataTuple(line_a), PipelineDataTuple(dim))
    #line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    #line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    #line_b = pipeline.append_step(D2VInterfaceWrapper.interface, PipelineDataTuple(line_b), PipelineDataTuple(dim))
    #line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_a, line_b), None)
    #line_ab = pipeline.append_step(Matcher.interface, PipelineDataTuple(line_ab),
    #                               PipelineDataTuple(model))
    #line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
#
    #configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
    #                              pipeline, src_properties, tgt_properties)
    #configuration_handler = ConfigurationHandler()
    #configuration_handler.execute(configuration)
#
#
#
    #name = "pseudod2v xgb"
    #pipeline = Pipeline()
    #line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    #line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    #line_a = pipeline.append_step(PseudoD2VInterfaceWrapper.interface, PipelineDataTuple(line_a), PipelineDataTuple(dim))
    #line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    #line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    #line_b = pipeline.append_step(PseudoD2VInterfaceWrapper.interface, PipelineDataTuple(line_b), PipelineDataTuple(dim))
    #line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_a, line_b), None)
    #line_ab = pipeline.append_step(Matcher.interface, PipelineDataTuple(line_ab),
    #                               PipelineDataTuple(model))
    #line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
#
    #configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
    #                              pipeline, src_properties, tgt_properties)
    #configuration_handler = ConfigurationHandler()
    #configuration_handler.execute(configuration)
#
#
#
#
#
#
#
    #name = "W2V_1 xgb"
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
    #line_ab = pipeline.append_step(Matcher.interface, PipelineDataTuple(line_ab),
    #                               PipelineDataTuple(model))
    #line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
#
    #configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
    #                              pipeline, src_properties, tgt_properties)
    #configuration_handler = ConfigurationHandler()
    #configuration_handler.execute(configuration)
#
#
#
#
#
#
#
    #name = "D2V_1 xgb"
    #pipeline = Pipeline()
    #line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    #line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    #line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    #line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    #line_ab = pipeline.append_step(D2V_1InterfaceWrapper.interface, PipelineDataTuple(line_a, line_b), PipelineDataTuple(dim))
    #line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(Matcher.interface, PipelineDataTuple(line_ab),
    #                               PipelineDataTuple(model))
    #line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
#
    #configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
    #                              pipeline, src_properties, tgt_properties)
    #configuration_handler = ConfigurationHandler()
    #configuration_handler.execute(configuration)
#
#
#
#
    #name = "pseudoD2V_1 xgb"
    #pipeline = Pipeline()
    #line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    #line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    #line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    #line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    #line_ab = pipeline.append_step(PseudoD2V_1InterfaceWrapper.interface, PipelineDataTuple(line_a, line_b), PipelineDataTuple(dim))
    #line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(Matcher.interface, PipelineDataTuple(line_ab),
    #                               PipelineDataTuple(model))
    #line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    #line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
#
    #configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
    #                              pipeline, src_properties, tgt_properties)
    #configuration_handler = ConfigurationHandler()
    #configuration_handler.execute(configuration)





if __name__ == '__main__':
    main()
