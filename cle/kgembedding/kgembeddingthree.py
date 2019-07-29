from skge import HolE, StochasticTrainer
from eval.evaldbkwik import generate_eval_files_dbkwik
# from https://github.com/mnick/scikit-kge
# with change from https://github.com/mnick/holographic-embeddings/issues/6
import numpy as np
import logging

def train(triples):
    # Load knowledge graph
    # N = number of entities
    # M = number of relations
    # xs = list of (subject, object, predicte) triples
    # ys = list of truth values for triples (1 = true, -1 = false)


    #N, M = 2,1
    #xs = [(0, 1, 0)]
    #ys = np.ones(len(xs))

    N, M, xs, ys = generate_kg_for_training(triples)

    # instantiate HolE with an embedding space of size 100
    model = HolE((N, N, M), 10)

    # instantiate trainer
    trainer = StochasticTrainer(model)

    # fit model to knowledge graph
    trainer.fit(xs, ys)

    model.E
    model.R
    CONFIGURATION.log('finish')


def generate_kg_for_training(triples):
    entity_id = 0
    entities = dict()
    relation_id = 0
    relations = dict()

    training_triples = list()
    for i, (s,p,o) in enumerate(triples):
        s_id = entities.get(s)
        if s_id is None:
            s_id = entity_id
            entities[s] = entity_id
            entity_id += 1

        o_id = entities.get(o)
        if o_id is None:
            o_id = entity_id
            entities[o] = entity_id
            entity_id += 1

        p_id = relations.get(p)
        if p_id is None:
            p_id = relation_id
            relations[p] = relation_id
            relation_id += 1

        training_triples.append((s_id, o_id, p_id))
        if i > 1000:
            break

    return entity_id, relation_id, np.array(training_triples), np.ones(len(training_triples))


#def count_entities_relations(triples):
#    entities, relations = set(), set()
#    for s, p, o in triples:
#        entities.add(s)
#        entities.add(o)
#        relations.add(p)
#    return len(entities), len(relations)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)
    logging.info("Start")
    #train(None)
    src_kg, dst_kg, gold_mapping = next(generate_eval_files_dbkwik())
    triples = src_kg.get_object_triples()
    #generate_kg_for_training(triples)
    train(triples)