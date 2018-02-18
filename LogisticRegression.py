import numpy as np


# part b and c
class LogisticRegression(object):
    def __init__(self, n_iter=10000, eta=0.1):
        self.n_iterations = n_iter
        self.eta = eta
        self.beta = []
        self.gradient_L = []

    def gradientAscent(self, X, y):
        X = self.fillBeta(X)
        self.beta = np.zeros(X.shape[1])
        for _ in range(self.n_iterations):
            # Gradient descent
            self.beta -= self.eta * np.dot(X.T, self.part_compute_L(X, y)) / len(y)
            self.gradient_L.append(np.absolute(self.part_compute_L(X, y)).sum())  # sigma
        print('Beta=', self.beta)

    def part_compute_L(self, X, y):
        return self.exponentBetaCmp(X) - y  # Fix required

    def exponentBetaCmp(self, X):
        return 1 / (1 + np.exp(-np.dot(X, self.beta)))

    def predict(self, X):
        X = self.fillBeta(X)
        return np.where(self.exponentBetaCmp(X) >= 0.5, 1, 0)

    def fillBeta(self, X):
        """
        One column gets added to X
        :param X: Matrix containing independent variables
        :return: X with one extra column
        """
        return np.concatenate([np.ones((len(X), 1)), X], axis=1)
