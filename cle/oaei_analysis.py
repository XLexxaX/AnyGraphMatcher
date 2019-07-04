
import os
from cle.configurations.Configuration import Configuration, InternalGoldStandard
from cle.configurations.ConfigurationHandler import ConfigurationHandler
from cle.configurations.PipelineTools import Pipeline, PipelineDataTuple
from cle.loadkg.loadWithRdflib import load_kg_with_rdflib_ttl_interface
from cle.graphdatatools import GraphToolbox
from cle.wordembedding import W2VInterfaceWrapper, D2VInterfaceWrapper, PseudoD2VInterfaceWrapper, \
    W2V_1InterfaceWrapper, D2V_1InterfaceWrapper, PseudoD2V_1InterfaceWrapper, SimpleTriplesEmbedder, \
    SimpleTriplesEmbedder_1, concat_combiner, ResourceRelationsEmbeddingWrapper, muse, SimpleLiteralsEmbedder_1, \
    WalkEmbedder_1, WalkD2V_1Embedder, RelationsWalkEmbedder_1
from cle.visualization import CategoriesVisualizer, StratifiedVisualizer, TypeVisualizer, FullVisualizer, \
    EmbeddingSaver, TSNEInterface
from cle.sentencegenerator import ReadSentencesInterfaceWrapper
from cle.matcher import FlatMatcher, StableRankMatcher, PredictionToXMLConverter, PureSyntaxMatcher, StableRankSyntaxMatcher
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from cle.StringMatching import StringMatcher_Interface

package_directory = os.path.dirname(os.path.abspath(__file__))

def main():



    #src_triples = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_full_strings',
    #                           'graph_triples_hilti_erp.nt')
    #tgt_triples = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_full_strings',
    #                           'graph_triples_hilti_web.nt')
    #src_corpus = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_full_strings',
    #                          'corpus_hilti_erp.txt')
    #tgt_corpus = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_full_strings',
    #                          'corpus_hilti_web.txt')
    #gold_mapping = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_full_strings',
    #                            'train_simple_sap_hilti.csv')
    src_triples = os.path.join(package_directory, '..', 'data', 'oaei_data',
                               'graph_triples_darkscape.nt')
    tgt_triples = os.path.join(package_directory, '..', 'data', 'oaei_data',
                               'graph_triples_oldschoolrunescape.nt')
    src_corpus = os.path.join(package_directory, '..', 'data', 'oaei_data',
                              'corpus_darkscape.txt')
    tgt_corpus = os.path.join(package_directory, '..', 'data', 'oaei_data',
                              'corpus_oldschoolrunescape.txt')
    gold_mapping = InternalGoldStandard({'trainsets':
                                            [os.path.join(package_directory, '..', 'data',
                                            'oaei_data', 'oaei_gold_standard2.csv')],
                                         'testsets':
                                             [os.path.join(package_directory, '..', 'data',
                                            'oaei_data', 'possible_matches.csv')]
                                        })
    dim = 2
    model = XGBClassifier()#LogisticRegression()
    labelfile = os.path.join(package_directory, '..', 'data', 'oaei_data','labels.txt')
    src_properties = StringMatcher_Interface.get_labels_from_file(labelfile)
    tgt_properties = StringMatcher_Interface.get_labels_from_file(labelfile)


    name = "OAEI_visualization"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    line_ab = pipeline.append_step(WalkEmbedder_1.interface, PipelineDataTuple(line_a, line_b),
                                   PipelineDataTuple(dim, 'steps', True, 1))
    line_ab = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(TSNEInterface.interface, PipelineDataTuple(line_ab), PipelineDataTuple(2))
    line_ab = pipeline.append_step(EmbeddingSaver.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(CategoriesVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(StratifiedVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(TypeVisualizer.interface, PipelineDataTuple(line_ab), None)
    line_ab = pipeline.append_step(FullVisualizer.interface, PipelineDataTuple(line_ab), None)


    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, dim,
                                  pipeline, src_properties, tgt_properties, calc_PLUS_SCORE=False, use_cache=False, use_streams=False)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)


    #line_ab = pipeline.append_step(WalkEmbedder_1.interface, PipelineDataTuple(line_a, line_b),
    #                               PipelineDataTuple(dim, 'steps', False, 3))
    #line_ab = pipeline.append_step(WalkEmbedder_1.interface, PipelineDataTuple(line_a, line_b),
    #                               PipelineDataTuple(dim, 'steps', True, 1))
    #line_ab = pipeline.append_step(WalkEmbedder_1.interface, PipelineDataTuple(line_a, line_b),
    #                               PipelineDataTuple(dim, 'batch', False, 1))
    #line_ab = pipeline.append_step(WalkEmbedder_1.interface, PipelineDataTuple(line_a, line_b),
    #                               PipelineDataTuple(100, 'batch', False, 1))

if __name__ == '__main__':
    main()
