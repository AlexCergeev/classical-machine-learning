import pandas as pd
import numpy as np
import random
from typing import Union, Callable

from sklearn.metrics import confusion_matrix


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class MyLogReg:
    """
    Linear Logistic Regression
    """
    def __init__(self, n_iter: int = 100, learning_rate: Union[float, Callable] = 0.01, 
                 weights: np.ndarray = None, metric: str = 'logloss', reg: str = None, 
                 l1_coef: float = 0, l2_coef: float = 0, sgd_sample: Union[int, float] = None, 
                 random_state: int = 42, eps: float = 1e-15, threshold=0.5):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.score = None
        self.eps = eps
        self.threshold = threshold
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.metrics_map = {
            "logloss": self.logloss,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc
        }

    def __str__(self):
        return "MyLogReg class: " + ", ".join(f"{k}={v}" for k, v in self.__dict__.items())

    def _add_const(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _sgd_idx(self, X):
        sgd_samples = self.sgd_sample
        if isinstance(self.sgd_sample, float):
            sgd_samples = int(self.sgd_sample * X.shape[0])
        elif self.sgd_sample is None:
            sgd_samples = X.shape[0]
            return list(range(X.shape[0]))
        return random.sample(range(X.shape[0]), sgd_samples)

    def _grad(self, X, y, idx):
        y_pred = sigmoid(X @ self.weights)
        grad = ((y_pred[idx] - y[idx]) @ X) / len(idx)
        if self.reg == "l1":
            grad += self.l1_coef * np.sign(self.weights)
        elif self.reg == "l2":
            grad += self.l2_coef * 2 * self.weights
        elif self.reg == "elasticnet":
            grad += self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights
        return grad

    def _loss(self, X, y):
        y_pred = sigmoid(X @ self.weights)
        loss = self.logloss(y, y_pred, self.eps)
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
        # random.seed(self.random_state)
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
        if self.metric in ['logloss', 'roc_auc']:
            self.score = self.metrics_map[self.metric](target, self.predict_proba(X))
        else:
            self.score = self.metrics_map[self.metric](target, self.predict(X))
        return self

    def get_coef(self):
        return self.weights[1:]

    def predict_proba(self, X):
        _X = self._add_const(X.to_numpy())
        return sigmoid(_X @ self.weights)
    
    def predict(self, X):
        return (self.predict_proba(X) > self.threshold).astype(int)

    @staticmethod
    def logloss(y_true, y_pred, eps=1e-15):
        """
        метод для получения значения локистической фунуции потерь 
        """
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
    
    @staticmethod
    def accuracy(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return (tp + tn) / (tp + fp + fn + tn)
    
    @staticmethod
    def precision(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fp)
    
    @staticmethod
    def recall(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn)
    
    @staticmethod
    def f1(y_true, y_pred, alpha=1):
        prec = precision(y_true, y_pred)
        rec = recall(y_true, y_pred)
        return (1 + alpha**2) * prec * rec / (alpha**2 * prec + rec)
    
    @staticmethod
    def roc_auc(y_true, y_pred):
        P = sum(y_true)
        N = len(y_true) - P 

        data = pd.concat([pd.Series(y_pred).rename('score'), 
                            pd.Series(y_true).rename('target')], axis=1)\
        .sort_values('score', ascending=False)
        data['dev'] = 0.
        for i, s, t, _ in data.itertuples():
            if not t:
                more = data[data['score']>s]['target'].sum()
                eq = data[data['score']==s]['target'].sum()/2
                data.loc[i, 'dev'] = more + eq
        roc_auc = data['dev'].sum() / (P * N)
        return roc_auc
        
    def get_best_score(self):
        return self.score

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

# model = MyLogReg(**{"n_iter": 50, "learning_rate": 0.1})
# model.fit(X, y, 10)
# print()
# print(model.get_coef().mean())