


import os
from cle.Configuration import Configuration
from cle.ConfigurationHandler import ConfigurationHandler
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression

package_directory = os.path.dirname(os.path.abspath(__file__))

def main():
    logfile = os.path.join(package_directory, '..', 'results.log')
    try:
        os.remove(logfile)
    except:
        pass

    ms = []
    ms = ms + [XGBClassifier()]
    #ms = ms + [make_pipeline(PolynomialFeatures(degree=3), Ridge())]

    for m in ms:

        name = "w2v / muse"
        src_corpus = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'graph_triples_Amazon1.nt')
        tgt_corpus = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'graph_triples_Google2.nt')
        src_triples = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'corpus_Amazon1.txt')
        tgt_triples = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'corpus_Google2.txt')
        gold_mapping = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'AmazonGoogleGold.csv')
        w2v = True
        d2v = False
        alignment = True
        matcher_model = m
        pca = False
        dim = 20
        configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, d2v, w2v, alignment, matcher_model, logfile, pca, dim)
        configuration_handler = ConfigurationHandler()
        configuration_handler.execute(configuration)

        name = "w2v"
        src_corpus = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'graph_triples_Amazon1.nt')
        tgt_corpus = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'graph_triples_Google2.nt')
        src_triples = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'corpus_Amazon1.txt')
        tgt_triples = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'corpus_Google2.txt')
        gold_mapping = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'AmazonGoogleGold.csv')
        w2v = True
        d2v = False
        alignment = False
        matcher_model = m
        pca = False
        dim = 20
        configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, d2v, w2v, alignment, matcher_model, logfile, pca, dim)
        configuration_handler = ConfigurationHandler()
        configuration_handler.execute(configuration)


        name = "w2v on literals / muse"
        src_corpus = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'graph_triples_Amazon1_2.nt')
        tgt_corpus = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'graph_triples_Google2_2.nt')
        src_triples = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'corpus_Amazon1_2.txt')
        tgt_triples = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'corpus_Google2_2.txt')
        gold_mapping = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'AmazonGoogleGold_2.csv')
        w2v = True
        d2v = False
        alignment = True
        matcher_model = m
        pca = False
        dim = 20
        configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, d2v, w2v, alignment, matcher_model, logfile, pca, dim)
        configuration_handler = ConfigurationHandler()
        configuration_handler.execute(configuration)

        name = "w2v on literals"
        src_corpus = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'graph_triples_Amazon1_2.nt')
        tgt_corpus = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'graph_triples_Google2_2.nt')
        src_triples = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'corpus_Amazon1_2.txt')
        tgt_triples = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'corpus_Google2_2.txt')
        gold_mapping = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'AmazonGoogleGold_2.csv')
        w2v = True
        d2v = False
        alignment = False
        matcher_model = m
        pca = False
        dim = 20
        configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, d2v, w2v, alignment, matcher_model, logfile, pca, dim)
        configuration_handler = ConfigurationHandler()
        configuration_handler.execute(configuration)


        name = "d2v / muse"
        src_corpus = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'graph_triples_Amazon1_2.nt')
        tgt_corpus = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'graph_triples_Google2_2.nt')
        src_triples = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'corpus_Amazon1_2.txt')
        tgt_triples = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'corpus_Google2_2.txt')
        gold_mapping = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'AmazonGoogleGold_2.csv')
        w2v = False
        d2v = True
        alignment = True
        matcher_model = m
        pca = False
        dim = 20
        configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, d2v, w2v, alignment, matcher_model, logfile, pca, dim)
        configuration_handler = ConfigurationHandler()
        configuration_handler.execute(configuration)

        name = "d2v"
        src_corpus = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'graph_triples_Amazon1_2.nt')
        tgt_corpus = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'graph_triples_Google2_2.nt')
        src_triples = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'corpus_Amazon1_2.txt')
        tgt_triples = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'corpus_Google2_2.txt')
        gold_mapping = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'AmazonGoogleGold_2.csv')
        w2v = False
        d2v = True
        alignment = False
        matcher_model = m
        pca = False
        dim = 20
        configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, d2v, w2v, alignment, matcher_model, logfile, pca, dim)
        configuration_handler = ConfigurationHandler()
        configuration_handler.execute(configuration)

        name = "w2v on literals / d2v / muse"
        src_corpus = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'graph_triples_Amazon1_2.nt')
        tgt_corpus = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'graph_triples_Google2_2.nt')
        src_triples = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'corpus_Amazon1_2.txt')
        tgt_triples = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'corpus_Google2_2.txt')
        gold_mapping = os.path.join(package_directory, '..', 'data', 'amazon_data', 'amazon_data', 'AmazonGoogleGold_2.csv')
        w2v = True
        d2v = True
        alignment = True
        matcher_model = m
        pca = False
        dim = 20
        configuration = Configuration(name, src_corpus, tgt_corpus, src_triples, tgt_triples, gold_mapping, d2v, w2v, alignment, matcher_model, logfile, pca, dim)
        configuration_handler = ConfigurationHandler()
        configuration_handler.execute(configuration)


if __name__ == '__main__':
    main()
