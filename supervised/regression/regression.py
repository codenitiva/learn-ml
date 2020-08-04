import numpy as np
import math


class BaseRegression(object):

    def __init__(self, n_iter, lr):
        self.n_iter = n_iter
        self.lr = lr
        self.w = []
        self.errors = []

    def init_weight(self, n_feat):
        bound = 1 / n_feat
        self.w = np.random.uniform(-bound, bound, (n_feat, ))

    def fit(self, X, y):

        X = np.insert(X, 0, 1, axis=1)
        self.init_weight(n_feat=X.shape[1])

        for it in range(self.n_iter):
            y_pred = X.dot(self.w)

            mse = np.mean(0.5 * (y_pred - y)**2)
            self.errors.append(mse)

            grad = (y_pred - y).dot(X)
            self.w -= self.lr * grad

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred


class LinearRegression(BaseRegression):

    def __init__(self, n_iter=1000, lr=0.01):
        super(LinearRegression, self).__init__(n_iter, lr)

    def fit(self, X, y):
        super(LinearRegression, self).fit(X, y)
