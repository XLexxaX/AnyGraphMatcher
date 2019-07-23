from sklearn.manifold import TSNE
import numpy as np


import numpy as np
from sklearn.manifold import TSNE

def test_tsne():
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    X_embedded = TSNE(n_components=2).fit_transform(X)
    print(X)
    print(X_embedded)