import pandas as pd
import numpy as np
import random
from typing import Union, Callable


class MyLineReg:
    """
    Linear Regression
    """
    def __init__(self, n_iter: int = 100, learning_rate: Union[float, Callable] = 0.01, 
                 weights: np.ndarray = None, metric: str = 'mse', reg: str = None, 
                 l1_coef: float = 0, l2_coef: float = 0, sgd_sample: Union[int, float] = None, 
                 random_state: int = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.metrics_map = {
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
            "mape": self.mape,
            "r2": self.r2
        }

    def __str__(self):
        return "MyLineReg class: " + ", ".join(f"{k}={v}" for k, v in self.__dict__.items())

    def _add_const(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _sgd_idx(self, X):
        sgd_samples = self.sgd_sample
        if isinstance(self.sgd_sample, float):
            sgd_samples = int(self.sgd_sample * X.shape[0])
        elif self.sgd_sample is None:
            sgd_samples = X.shape[0]
        return random.sample(range(X.shape[0]), sgd_samples)

    def _grad(self, X, y, idx):
        y_pred = X @ self.weights
        grad = 2 * (X[idx].T @ (y_pred[idx] - y[idx])) / len(idx)
        if self.reg == "l1":
            grad += self.l1_coef * np.sign(self.weights)
        elif self.reg == "l2":
            grad += self.l2_coef * 2 * self.weights
        elif self.reg == "elasticnet":
            grad += self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights
        return grad

    def _loss(self, X, y):
        y_pred = X @ self.weights
        loss = self.mse(y, y_pred)
        if self.reg == "l1":
            loss += self.l1_coef * np.sum(np.abs(self.weights))
        elif self.reg == "l2":
            loss += self.l2_coef * np.sum(self.weights ** 2)
        elif self.reg == "elasticnet":
            loss += self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum(self.weights ** 2)
        return loss

    def _lr(self, iter):
        if isinstance(self.learning_rate, float):
            return self.learning_rate
        return self.learning_rate(iter)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = False):
        random.seed(self.random_state)
        features = self._add_const(X.to_numpy())
        target = y.to_numpy()
        self.weights = np.ones(features.shape[1])

        for i in range(1, self.n_iter + 1):
            idx = self._sgd_idx(features)
            grad = self._grad(X=features, y=target, idx=idx)
            self.weights -= self._lr(i) * grad

            if verbose and i % verbose == 0:
                loss = self._loss(X=features, y=target)
                print(f"Iteration {i}: loss = {loss:.4f}")

        self.score = self.metrics_map[self.metric](target, features @ self.weights)

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X):
        _X = self._add_const(X.to_numpy())
        return _X @ self.weights

    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(MyLineReg.mse(y_true, y_pred))

    @staticmethod
    def mape(y_true, y_pred):
        return 100 * np.mean(np.abs((y_true - y_pred) / y_true))

    @staticmethod
    def r2(y_true, y_pred):
        return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    def get_best_score(self):
        return self.score
