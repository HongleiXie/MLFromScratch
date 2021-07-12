# coding: utf-8
# Implement the K Nearest Neighbors from scratch
import numpy as np
from scipy.spatial.distance import euclidean, cosine
from collections import Counter
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split

# a.sort(): in-place a
# sorted(a, key=lambda x: x[0]): not changing a, returning a new a, sorted by the key

class KNNBase(object):
    """
    O(N) operation where the N is the size of the corpus
    Check out Locality-Sensitive Hashing for Faster KNN.
    Each vector is hashed multiple times, typically on the order of dozens to hundreds.
    """
    def __init__(self, k=5, distance_func=cosine):
        self.k = k
        self.distance_func = distance_func

    def _vote(self, neighbors_targets):
        return NotImplementedError()

    def _predict(self, X_train, X_test, y_train):

        y_pred = np.zeros(X_test.shape[0], dtype=y_train.dtype)

        for i, test_sample in enumerate(X_test):
            idx = np.argsort(self.distance_func(test_sample, x) for x in X_train)[:self.k]
            neighbors_targets = np.array([y_train[i] for i in idx])
            y_pred[i] = self._vote(neighbors_targets)

        return y_pred


class KNNClassifier(KNNBase):
    def _vote(self, neighbors_targets):
        return Counter(neighbors_targets).most_common(1)[0][0]

class KNNRegressor(KNNBase):
    def _vote(self, neighbors_targets):
        return np.mean(neighbors_targets)

if __name__ == '__main__':

    iris = load_iris()
    # we only take the first two features
    X = iris.data[:, :2]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    print(KNNClassifier(k=10)._predict(X_train=X_train, y_train=y_train, X_test = X_test))

    boston = load_boston()
    X = boston['data']
    y = boston['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    print(KNNRegressor(k=10)._predict(X_train=X_train, y_train=y_train, X_test=X_test))