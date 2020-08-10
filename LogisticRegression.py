# coding: utf-8
# Implement the Logistic Regression from scratch
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import euclidean

# GD update rule: para_new = para_old - learning_rate * gradient
class Sigmoid():
    def __call__(self, x):
        #allows the class's instance to be called as a function, not always modifying the instance itself.
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

# s = Sigmoid()
# s()

class LogisticRegression(object):
    def __init__(self, learning_rate, max_iter=100, tot=0.1):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tot = tot
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X):
        """
        :param X: shape (m, p)
        """
        n_features = X.shape[1]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, n_features) # shape: (n_features, )

    def _fit(self, X, y):
        self._initialize_parameters(X)
        num_iter = 0
        diff_param = 999
        while (diff_param > self.tot and num_iter < self.max_iter):
            num_iter += 1
            y_pred = self.sigmoid(X.dot(self.param))
            grad = (y_pred - y).dot(X) # shape: (n_features, )
            old_para = self.param
            new_para = old_para - self.lr * grad
            diff_param = euclidean(old_para, new_para)
            self.param = new_para

    def _predict(self, X):
        y_pred = self.sigmoid(X.dot(self.param))
        return y_pred


if __name__ == '__main__':
    breast_cancer = load_breast_cancer()
    X = breast_cancer['data']
    y = breast_cancer['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = LogisticRegression(learning_rate=0.01, max_iter=200, tot=0.001)
    clf._fit(X_train, y_train)
    pred = clf._predict(X_test)
    print(roc_auc_score(y_test, pred))
