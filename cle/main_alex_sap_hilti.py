import os
from configurations.Configuration import Configuration
from configurations.ConfigurationHandler import ConfigurationHandler
from configurations.PipelineTools import Pipeline, PipelineDataTuple
from loadkg.loadFromXml import load_kg_from_xml_interface
from loadkg.loadWithRdflib import load_kg_with_rdflib_ttl_interface
from graphdatatools import GraphToolbox
from wordembedding import W2VInterfaceWrapper, D2VInterfaceWrapper, concat_combiner
from sentencegenerator import ReadSentencesInterfaceWrapper
from wordembedding import muse
from matcher import FlatMatcher
from xgboost import XGBClassifier

package_directory = os.path.dirname(os.path.abspath(__file__))

def main():
    logfile = os.path.join(package_directory, '..', 'results.log')
    try:
        os.remove(logfile)
    except:
        pass

    src_triples = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_full_strings',
                               'graph_triples_hilti_erp.nt')
    tgt_triples = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_full_strings',
                               'graph_triples_hilti_web.nt')
    src_corpus = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_full_strings',
                              'corpus_hilti_erp.txt')
    tgt_corpus = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_full_strings',
                              'corpus_hilti_web.txt')
    gold_mapping = os.path.join(package_directory, '..', 'data', 'sap_hilti_data', 'sap_hilti_full_strings',
                                'sap_hilti_gold.csv')
    dim = 20
    model = XGBClassifier()

    name = "w2v d2v concat muse xgb"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_a),
                                  PipelineDataTuple(src_corpus))
    line_a = pipeline.append_step(W2VInterfaceWrapper.interface, PipelineDataTuple(line_a), PipelineDataTuple(dim))
    line_a = pipeline.append_step(D2VInterfaceWrapper.interface, PipelineDataTuple(line_a), PipelineDataTuple(dim))
    line_a = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_a), None)
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_b),
                                  PipelineDataTuple(tgt_corpus))
    line_b = pipeline.append_step(W2VInterfaceWrapper.interface, PipelineDataTuple(line_b), PipelineDataTuple(dim))
    line_b = pipeline.append_step(D2VInterfaceWrapper.interface, PipelineDataTuple(line_b), PipelineDataTuple(dim))
    line_b = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_b), None)
    line_ab = pipeline.append_step(muse.interface, PipelineDataTuple(line_a, line_b), PipelineDataTuple(gold_mapping))
    line_ab = pipeline.append_step(FlatMatcher.interface, PipelineDataTuple(line_ab),
                                   PipelineDataTuple(gold_mapping, model, logfile, name))

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, logfile, dim,
                                  pipeline)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)





    name = "w2v d2v concat xgb"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_a),
                                  PipelineDataTuple(src_corpus))
    line_a = pipeline.append_step(W2VInterfaceWrapper.interface, PipelineDataTuple(line_a), PipelineDataTuple(dim))
    line_a = pipeline.append_step(D2VInterfaceWrapper.interface, PipelineDataTuple(line_a), PipelineDataTuple(dim))
    line_a = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_a), None)
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(ReadSentencesInterfaceWrapper.interface, PipelineDataTuple(line_b),
                                  PipelineDataTuple(tgt_corpus))
    line_b = pipeline.append_step(W2VInterfaceWrapper.interface, PipelineDataTuple(line_b), PipelineDataTuple(dim))
    line_b = pipeline.append_step(D2VInterfaceWrapper.interface, PipelineDataTuple(line_b), PipelineDataTuple(dim))
    line_b = pipeline.append_step(concat_combiner.interface, PipelineDataTuple(line_b), None)
    line_ab = pipeline.append_step(FlatMatcher.interface, PipelineDataTule(line_a, line_b),
                                   PipelineDataTuple(gold_mapping, model, logfile, name))

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, logfile, dim,
                                  pipeline)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)





    name = "w2v muse xgb"
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
    line_ab = pipeline.append_step(muse.interface, PipelineDataTuple(line_a, line_b), PipelineDataTuple(gold_mapping))
    line_ab = pipeline.append_step(FlatMatcher.interface, PipelineDataTuple(line_ab),
                                   PipelineDataTuple(gold_mapping, model, logfile, name))

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, logfile, dim,
                                  pipeline)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)





    name = "w2v xgb"
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
    line_ab = pipeline.append_step(FlatMatcher.interface, PipelineDataTuple(line_a, line_b),
                                   PipelineDataTuple(gold_mapping, model, logfile, name))

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, logfile, dim,
                                  pipeline)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)





    name = "d2v muse xgb"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(D2VInterfaceWrapper.interface, PipelineDataTuple(line_a), PipelineDataTuple(dim))
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(D2VInterfaceWrapper.interface, PipelineDataTuple(line_b), PipelineDataTuple(dim))
    line_ab = pipeline.append_step(muse.interface, PipelineDataTuple(line_a, line_b), PipelineDataTuple(gold_mapping))
    line_ab = pipeline.append_step(FlatMatcher.interface, PipelineDataTuple(line_ab),
                                   PipelineDataTuple(gold_mapping, model, logfile, name))

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, logfile, dim,
                                  pipeline)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)




    name = "d2v xgb"
    pipeline = Pipeline()
    line_a = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_a), PipelineDataTuple(src_triples))
    line_a = pipeline.append_step(D2VInterfaceWrapper.interface, PipelineDataTuple(line_a), PipelineDataTuple(dim))
    line_b = pipeline.append_step(load_kg_with_rdflib_ttl_interface, None, PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(GraphToolbox.interface, PipelineDataTuple(line_b), PipelineDataTuple(tgt_triples))
    line_b = pipeline.append_step(D2VInterfaceWrapper.interface, PipelineDataTuple(line_b), PipelineDataTuple(dim))
    line_ab = pipeline.append_step(FlatMatcher.interface, PipelineDataTuple(line_a, line_b),
                                   PipelineDataTuple(gold_mapping, model, logfile, name))

    configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, logfile, dim,
                                  pipeline)
    configuration_handler = ConfigurationHandler()
    configuration_handler.execute(configuration)


if __name__ == '__main__':
    main()
