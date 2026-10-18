# Implement the six required course methods here, using NumPy:
# mean_squared_error_gd, mean_squared_error_sgd, least_squares,
# ridge_regression, logistic_regression, and reg_logistic_regression.
# Preserve the required signatures and return the final (w, loss).
# Returned losses exclude regularization penalties. Keep this file at repo root.


# Create a virtual class of linear models

import numpy as np
from abc import ABC, abstractmethod


class LinMod(ABC):
    """Generic linear model with methods to implment in each case and common methods"""

    # declare and initialize attributes
    def __init__(self, d):
        self.w = np.zeros(d)

    # to implement in each sub class
    @abstractmethod
    def loss(self, y, x):
        raise NotImplementedError

    @abstractmethod
    def gradient(self, y, x):
        raise NotImplementedError

    # common methods
    def predict(self, x):
        return x @ self.w

    def gd(self, y, x, w0, max_it, gamma):
        self.w = w0.copy()
        for i in range(max_it):
            self.w -= gamma * self.gradient(y, x)
        return self.w, self.loss(y, x)

    def sgd(self, y, x, w0, max_it, gamma):
        # np.random.seed(42) for tests
        self.w = w0.copy()
        n = len(y)
        for i in range(max_it):
            j = np.random.randint(n)
            # y[j:j+1], x[j:j+1] to keep shape (1,) and (1, d) otherwise doesn't work in gradient
            self.w -= gamma * self.gradient(y[j : j + 1], x[j : j + 1])
        return self.w, self.loss(y, x)


class LinearMSE(LinMod):
    """Linear regression with loss = 0.5MSE"""

    # implement virtual methods
    def loss(self, y, x):
        err = y - self.predict(x)
        return 0.5 * np.mean(err**2)

    def gradient(self, y, x):
        err = y - self.predict(x)
        return -(x.T @ err) / len(y)

    def closed_sol(self, y, x, lmbd=0.0):
        A = x.T @ x + lmbd * np.eye(x.shape[1])
        b = x.T @ y
        self.w = np.linalg.solve(A, b)
        return self.w, self.loss(y, x)


class RidgeMSE(LinearMSE):
    # need to overload __init__ to include lambda

    def __init__(self, d, lmbd):
        super().__init__(d)  # call d initialization of mother class
        self.lmbd = lmbd  # init new attribute lambda

    def gradient(self, y, x):
        err = y - self.predict(x)
        return super().gradient(y, x) + 2 * self.lmbd * self.w  # Call mother gradient


class Logistic(LinMod):

    # overload predict of linear model, since it is classification, to return a proba
    def predict(self, x):
        return 1.0 / (1.0 + np.exp(-(x @ self.w)))

    def loss(self, y, x):
        z = self.predict(x)
        return np.mean(np.logaddexp(0, z) - y * z)

    def gradient(self, y, x):
        z = self.predict(x)
        p = 1.0 / (1.0 + np.exp(-z))
        return x.T @ (p - y) / len(y)


class RegLog(Logistic):
    # need to overload __init__ to include lambda

    def __init__(self, d, lmbd):
        super().__init__(d)  # call d initialization of mother class
        self.lmbd = lmbd  # init new attribute lambda

    def gradient(self, y, x):
        g = super().gradient(y, x)  # call mother gradient
        return g + 2 * self.lmbd * self.w


###########################################################################################################


# implementation of the 6 functions


def mean_squared_error_gd(y, x, w0, max_it, gamma):
    """Linear regression via batch gradient descent.
    Minimizes 0.5 * MSE over all samples and returns (w, MSE_loss).
    """
    model = LinearMSE(x.shape[1])
    return model.gd(y, x, w0, max_it, gamma)


def mean_squared_error_sgd(y, x, w0, max_it, gamma):
    """Linear regression via stochastic gradient descent (mini-batch = 1).
    Samples one datapoint per iteration and returns (w, MSE_loss).
    """
    model = LinearMSE(x.shape[1])
    return model.sgd(y, x, w0, max_it, gamma)


def least_squares(y, x):
    """Least squares regression via normal equations.
    Solves (X^T X) w = X^T y and returns (w, MSE_loss).
    """
    model = LinearMSE(x.shape[1])
    return model.closed_sol(y, x)


def ridge_regression(y, x, lmbd):
    """Ridge regression via normal equations.
    Solves (X^T X + 2n*lambda*I) w = X^T y and returns (w, MSE_loss).
    Note: the returned loss does NOT include the penalty term.
    """
    model = RidgeMSE(x.shape[1], lmbd)
    return model.closed_sol(y, x, lmbd)


def logistic_regression(y, x, w0, max_it, gamma):
    """Logistic regression via gradient descent (y in {0, 1}).
    Minimizes the negative log-likelihood and returns (w, logistic_loss).
    """
    model = Logistic(x.shape[1])
    return model.gd(y, x, w0, max_it, gamma)


def reg_logistic_regression(y, x, lmbd, w0, max_it, gamma):
    """Regularized logistic regression via gradient descent (y in {0, 1}).
    Minimizes negative log-likelihood with penalty lambda*||w||^2 and
    returns (w, logistic_loss) where the loss does NOT include the penalty.
    """
    model = RegLog(x.shape[1], lmbd)
    return model.gd(y, x, w0, max_it, gamma)
