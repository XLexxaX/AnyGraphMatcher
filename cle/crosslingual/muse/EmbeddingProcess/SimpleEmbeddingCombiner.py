import numpy as np

def combine_fixed_size(embeddings, emb_size):
    e = list()
    for n1 in embeddings:
        for n2 in n1:
            if (emb_size<=len(e)):
                return e
            e = e + [n2]
    return e + list(np.zeros(emb_size-len(e)))



def combine(embeddings):
    e = list()
    for n1 in embeddings:
        for n2 in n1:
            e = e + [n2]
    return e
