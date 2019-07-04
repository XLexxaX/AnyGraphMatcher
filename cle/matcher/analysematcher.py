import logging
from cle.matcher.basematcher import BaseMatcher
from cle.crosslingual.ccaprojection import cca_projection

logger = logging.getLogger(__name__)


class AnalyseMatcher(BaseMatcher):

    def __init__(self, embedding_generation_function):
        self.embedding_generation_function = embedding_generation_function
        self.src_proj_embedding = None
        self.dst_proj_embedding = None

    def compute_mapping(self):
        src_embedding = self.embedding_generation_function(self.src_kg)
        dst_embedding = self.embedding_generation_function(self.dst_kg)


        diff_vector_list = []
        for src, dst, rel, confidence in self.initial_mapping:
            if src in src_embedding and dst in dst_embedding:
                diff_vector = src_embedding[src] - model[target]
                diff_vector_list.append(diff_vector)


            yield (src, dst)


        list_of_source_target = [(source, target) for source, target in source_target if
                                 source in model and target in model]

        diff_vector_list = []
        label = []
        for source, target in list_of_source_target:
            diff_vector = model[source] - model[target]
            diff_vector_list.append(diff_vector)
            label.append(target.replace('http://dbpedia.org/resource/', ''))
        test = np.array(diff_vector_list)

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(test)

        fig, ax = plt.subplots()
        ax.scatter(principalComponents[:, 0], principalComponents[:, 1])
        blub = principalComponents.shape[0]
        for i in range(principalComponents.shape[0]):
            ax.annotate(label[i], (principalComponents[i, 0], principalComponents[i, 1]))
        plt.show()

        test = model.most_similar('http://dbpedia.org/resource/Half-blood', topn=50)
        testtwo = model.most_similar('http://dbpedia.org/resource/Harry_Potter', topn=50)

        source, target = list_of_source_target[0]
        source_test, target_test = list_of_source_target[1]
        # result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
        result = model.most_similar(positive=[source_test, target], negative=[source], topn=50)
        print("test")










        src_embedding


        self.src_proj_embedding, self.dst_proj_embedding = cca_projection(
            src_embedding, dst_embedding, self.generate_lexicon_from_initial_mapping(), self.top_correlation_ratio)

    def get_mapping(self):
        mapping = set()

        return mapping

    def get_mapping_with_ranking(self, elements, topn=20):
        results = []
        for e in elements:
            if e not in self.src_proj_embedding:
                results.append([])
            else:
                results.append(self.dst_proj_embedding.most_similar(positive=[self.src_proj_embedding[e]], topn=topn))
        return results
