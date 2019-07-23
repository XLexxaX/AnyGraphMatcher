import os
from configurations.Configuration import Configuration
from configurations.ConfigurationHandler import ConfigurationHandler
from configurations.PipelineTools import Pipeline, PipelineDataTuple
from loadkg.loadWithRdflib import load_kg_with_rdflib_ttl_interface
from graphdatatools import GraphToolbox
from wordembedding import concat_combiner, muse, W2V_1InterfaceWrapper
from visualization import CategoriesVisualizer, TypeVisualizer, StratifiedVisualizer, FullVisualizer
from matcher import EmbeddingMatcher, Matcher
from sentencegenerator import ReadSentencesInterfaceWrapper

#from xgboost import XGBClassifier

package_directory = os.path.dirname(os.path.abspath(__file__))

def main():


    src_triples = os.path.join(package_directory, '..', 'data', 'oaei_data',
                               'graph_triples_darkscape.nt')
    tgt_triples = os.path.join(package_directory, '..', 'data', 'oaei_data',
                               'graph_triples_oldschoolrunescape.nt')
    src_corpus = os.path.join(package_directory, '..', 'data', 'oaei_data',
                              'corpus_darkscape.txt')
    tgt_corpus = os.path.join(package_directory, '..', 'data', 'oaei_data',
                              'corpus_oldschoolrunescape.txt')
    gold_mapping = os.path.join(package_directory, '..', 'data', 'oaei_data',
                                'gold_standard.csv')

    dim = 3
    #model = make_pipeline(PolynomialFeatures(6), Ridge())#DecisionTreeClassifier() #make_pipeline(PolynomialFeatures(8), Ridge())
    #model = sklearn.linear_model.LinearRegression()
    #from sklearn.ensemble import RandomForestRegressor
    #model = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
    #model = LinearSVC(C=0.01, class_weight=None, dual=True, fit_intercept=True,
    #                  intercept_scaling=1, loss='squared_hinge', max_iter=1000,
    #                  multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    src_properties = ["http://rdata2graph.sap.com/hilti_erp/property/mara_fert.maktx"]
    tgt_properties = ["http://rdata2graph.sap.com/hilti_web/property/products.name"]



    name = "W2V_1 muse xgb with 50k only on embeddings"
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
    line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(muse.interface, PipelineDataTuple(line_ab), PipelineDataTuple(gold_mapping))
    line_ab = pipeline.append_step(EmbeddingMatcher.interface, PipelineDataTuple(line_ab),
                                   PipelineDataTuple(model))
    line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(FullVisualizer.interface, PipelineDataTuple(line_ab), None)

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
                                  pipeline, src_properties, tgt_properties)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)




    name = "W2V_1 muse xgb with 50k on embeddings and sim"
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
    line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(muse.interface, PipelineDataTuple(line_ab), PipelineDataTuple(gold_mapping))
    line_ab = pipeline.append_step(EmbeddingMatcher.interface, PipelineDataTuple(line_ab),
                                   PipelineDataTuple(model))
    line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(FullVisualizer.interface, PipelineDataTuple(line_ab), None)

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
                                  pipeline, src_properties, tgt_properties)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)



if __name__ == '__main__':
    main()
