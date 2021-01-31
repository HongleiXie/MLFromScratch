# coding: utf-8
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class Kmeans(object):
    """
    Implement the most vanilla K-means algorithm from scratch
    """
    def __init__(self, k: int, random_seed: int=2, max_iter: int=100) -> None:
        self.k = k
        self.seed = random_seed
        self.max_iter = max_iter

    def _init_centroids(self, X: np.array) -> np.array:
        rng = np.random.RandomState(self.seed) # fix the seed
        i = rng.permutation(X.shape[0])[:self.k] # shuffle the rows in X and pick the first k rows
        centers = X[i]
        return centers

    def _label_clusters(self, centers: np.array, X: np.array) -> np.array:
        labels = pairwise_distances_argmin(X, centers) # computes for each row in X, the index of the row of Y which is closest
        return labels

    def _update_centroids(self, labels: np.array, X: np.array) -> np.array:
        new_centers = np.array([X[labels == i].mean(0) for i in range(self.k)])
        return new_centers

    def predict(self, X: np.array) -> Tuple[np.array, np.array]:
        """
        Perform K-means
        """

        # init
        centers = self._init_centroids(X)
        labels = np.zeros(X.shape[0])
        # need to scale the X before K-means
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        for _ in range(self.max_iter):
            labels = self._label_clusters(centers, X)
            new_centers = self._update_centroids(labels, X)
            if np.all(centers == new_centers):
                break
            centers = new_centers

        return centers, labels

if __name__ == '__main__':
    X = np.random.rand(100, 3)
    km = Kmeans(k=2)
    km.predict(X)


