# Implement the six required course methods here, using NumPy:
# mean_squared_error_gd, mean_squared_error_sgd, least_squares,
# ridge_regression, logistic_regression, and reg_logistic_regression.
# Preserve the required signatures and return the final (w, loss).
# Returned losses exclude regularization penalties. Keep this file at repo root.


# Create a virtual class of linear models

import numpy as np
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


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


class SVM(LinMod):
    """Soft support vector machine model"""
    def __init__(self, d, lmbd):
        super().__init__(d)  # call d initialization of mother class
        self.lmbd = lmbd  # init new attribute lambda

    def loss(self, y, x):
        err = 1 - y*self.predict(x)
        return np.mean(np.maximum(0, err))

    def gradient(self, y, x):
        err = 1 - y*self.predict(x)
        mask = (err > 0).astype(float)      # on the margin is not an error
        return self.lmbd * self.w -(x.T @ (y * mask)) / len(y)


class LinearMSE(LinMod):
    """Linear regression with loss = 0.5MSE"""

    # implement virtual methods
    def loss(self, y, x):
        err = y - self.predict(x)
        return 0.5 * np.mean(err**2)

    def gradient(self, y, x):
        err = y - self.predict(x)
        return -(x.T @ err) / len(y)

    def closed_sol(self, y, x):
        A = x.T @ x
        b = x.T @ y
        self.w = np.linalg.solve(A, b)
        return self.w, self.loss(y, x)


class RidgeMSE(LinearMSE):
    # need to overload __init__ to include lambda

    def __init__(self, d, lmbd):
        super().__init__(d)  # call d initialization of mother class
        self.lmbd = lmbd  # init new attribute lambda

    def gradient(self, y, x):
        return super().gradient(y, x) + 2 * self.lmbd * self.w  # Call mother gradient

    # need to overload closed solution to apply penalty with factor 2n
    def closed_sol(self, y, x):
        n = len(y)
        A = x.T @ x + 2 * n * self.lmbd * np.eye(x.shape[1])
        b = x.T @ y
        self.w = np.linalg.solve(A, b)
        return self.w, self.loss(y, x)


class Logistic(LinMod):

    # overload predict of linear model, since it is classification, to return a proba
    def predict(self, x):
        return 1.0 / (1.0 + np.exp(-(x @ self.w)))

    def loss(self, y, x):
        z = x @ self.w
        return np.mean(np.logaddexp(0, z) - y * z)

    def gradient(self, y, x):
        z = x @ self.w
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
    return model.closed_sol(y, x)


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


def svm(y, x, lmbd, w0, max_it, gamma):
    """Soft-margin SVM via gradient descent (y in {-1, +1}).
    Minimizes hinge loss + λ/2 * ||w||² and returns (w, hinge_loss).
    The returned loss does NOT include the penalty term.
    """
    model = SVM(x.shape[1], lmbd)
    return model.gd(y, x, w0, max_it, gamma)


###########################################################################################################
#                                                                                                         #
#                    SKLEARN-COMPATIBLE WRAPPERS FOR THE CUSTOM MODELS                                    #
#                                                                                                         #
###########################################################################################################


class _BaseCustomClassifier(BaseEstimator, ClassifierMixin):
    """Base wrapper for custom classifiers (logistic, reg-logistic, svm).

    Handles:
      - input validation (check_X_y, check_array)
      - class label decoding to {0, 1} internal representation
      - intercept column injection (adds column of 1s)
      - predict / predict_proba consistency with sklearn API
    """

    def _validate_and_prepare(self, X, y=None):
        if y is None:
            X = check_array(X, dtype=np.float64, ensure_min_samples=1)
        else:
            X, y = check_X_y(X, y, dtype=np.float64, ensure_min_samples=1)
        # Add intercept column
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_aug, y

    def _augment(self, X):
        X = check_array(X, dtype=np.float64, ensure_min_samples=1)
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _check_fitted(self):
        check_is_fitted(self, attributes=["w_", "classes_"])


class CustomLogisticRegression(_BaseCustomClassifier):
    """sklearn wrapper around `logistic_regression` (gradient descent)."""

    def __init__(self, max_it=1000, gamma=0.1, fit_intercept=True):
        self.max_it = max_it
        self.gamma = gamma
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"CustomLogisticRegression only supports binary classification, "
                f"got classes {self.classes_}"
            )

        if self.fit_intercept:
            X_aug, y_arr = self._validate_and_prepare(X, y)
        else:
            X_aug = check_array(X, dtype=np.float64)
            y_arr = np.asarray(y)

        # Map labels to {0, 1}
        y_internal = (y_arr == self.classes_[1]).astype(np.float64)

        w0 = np.zeros(X_aug.shape[1])
        self.w_, self.loss_ = logistic_regression(
            y_internal, X_aug, w0, self.max_it, self.gamma
        )
        return self

    def predict_proba(self, X):
        self._check_fitted()
        X_aug = self._augment(X) if self.fit_intercept else check_array(X)
        z = X_aug @ self.w_
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = (proba[:, 1] > 0.5).astype(int)
        return self.classes_[idx]


class CustomRegLogisticRegression(_BaseCustomClassifier):
    """sklearn wrapper around `reg_logistic_regression` (gradient descent + L2)."""

    def __init__(self, lmbd=0.1, max_it=1000, gamma=0.1, fit_intercept=True):
        self.lmbd = lmbd
        self.max_it = max_it
        self.gamma = gamma
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"CustomRegLogisticRegression only supports binary classification, "
                f"got classes {self.classes_}"
            )

        if self.fit_intercept:
            X_aug, y_arr = self._validate_and_prepare(X, y)
        else:
            X_aug = check_array(X, dtype=np.float64)
            y_arr = np.asarray(y)

        y_internal = (y_arr == self.classes_[1]).astype(np.float64)

        w0 = np.zeros(X_aug.shape[1])
        self.w_, self.loss_ = reg_logistic_regression(
            y_internal, X_aug, self.lmbd, w0, self.max_it, self.gamma
        )
        return self

    def predict_proba(self, X):
        self._check_fitted()
        X_aug = self._augment(X) if self.fit_intercept else check_array(X)
        z = X_aug @ self.w_
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = (proba[:, 1] > 0.5).astype(int)
        return self.classes_[idx]


class CustomSVM(_BaseCustomClassifier):
    """sklearn wrapper around `svm` (gradient descent, hinge loss).

    Internal convention: y in {-1, +1}.
    sklearn convention:    y in {0, 1} (or arbitrary labels).
    This wrapper handles the mapping transparently.
    """

    def __init__(self, lmbd=0.1, max_it=1000, gamma=0.1, fit_intercept=True):
        self.lmbd = lmbd
        self.max_it = max_it
        self.gamma = gamma
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"CustomSVM only supports binary classification, "
                f"got classes {self.classes_}"
            )

        if self.fit_intercept:
            X_aug, y_arr = self._validate_and_prepare(X, y)
        else:
            X_aug = check_array(X, dtype=np.float64)
            y_arr = np.asarray(np.where(y == self.classes_[1], 1, -1))

        # Map {0, 1} → {-1, +1} for the SVM gradient
        y_internal = 2 * (y_arr == self.classes_[1]).astype(np.float64) - 1

        w0 = np.zeros(X_aug.shape[1])
        
        self.w_, self.loss_ = svm(y_internal, X_aug, self.lmbd, w0, self.max_it, self.gamma)
        return self

    def decision_function(self, X):
        """Return raw scores w^T x (useful for AUC / thresholds)."""
        self._check_fitted()
        X_aug = self._augment(X) if self.fit_intercept else check_array(X)
        return X_aug @ self.w_

    def predict(self, X):
        scores = self.decision_function(X)
        idx = (scores > 0).astype(int)
        return self.classes_[idx]







class CustomLinearRegression(BaseEstimator, RegressorMixin):
    """sklearn wrapper around `least_squares` / `mean_squared_error_gd` / `mean_squared_error_sgd`.

    Parameters
    ----------
    method : str
        One of "closed_form" (least_squares), "gd" (batch GD), "sgd" (stochastic GD).
    max_it : int
        Number of iterations (ignored for "closed_form").
    gamma : float
        Learning rate (ignored for "closed_form").
    fit_intercept : bool
        Whether to add a column of ones (equivalent to an intercept).
    """

    _VALID_METHODS = {"closed_form", "gd", "sgd"}

    def __init__(self, method="closed_form", max_it=1000, gamma=0.1,
                 fit_intercept=True):
        self.method = method
        self.max_it = max_it
        self.gamma = gamma
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.method not in self._VALID_METHODS:
            raise ValueError(
                f"method must be one of {self._VALID_METHODS}, got {self.method!r}"
            )
        X, y = check_X_y(X, y, dtype=np.float64, ensure_min_samples=1)

        X_aug = (
            np.hstack([np.ones((X.shape[0], 1)), X])
            if self.fit_intercept else X
        )
        w0 = np.zeros(X_aug.shape[1])

        if self.method == "closed_form":
            self.w_, self.loss_ = least_squares(y, X_aug)
        elif self.method == "gd":
            self.w_, self.loss_ = mean_squared_error_gd(
                y, X_aug, w0, self.max_it, self.gamma
            )
        else:  # sgd
            self.w_, self.loss_ = mean_squared_error_sgd(
                y, X_aug, w0, self.max_it, self.gamma
            )
        return self

    def predict(self, X):
        check_is_fitted(self, attributes=["w_"])
        X = check_array(X, dtype=np.float64)
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X]) if self.fit_intercept else X
        return X_aug @ self.w_


class CustomRidgeRegression(BaseEstimator, RegressorMixin):
    """sklearn wrapper around `ridge_regression` (closed-form ridge)."""

    def __init__(self, lmbd=0.1, fit_intercept=True):
        self.lmbd = lmbd
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64, ensure_min_samples=1)
        X_aug = (
            np.hstack([np.ones((X.shape[0], 1)), X])
            if self.fit_intercept else X
        )
        self.w_, self.loss_ = ridge_regression(y, X_aug, self.lmbd)
        return self

    def predict(self, X):
        check_is_fitted(self, attributes=["w_"])
        X = check_array(X, dtype=np.float64)
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X]) if self.fit_intercept else X
        return X_aug @ self.w_