import pandas as pd
import numpy as np

from sklearn.datasets import make_regression

class MyLineReg():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.metrics = None
        self.reg = None
        self.l1_coef = 0
        self.l2_coef = 0
        self._X = None
        self._y = None
        for key, value in self.kwargs.items():
            setattr(self, key, value)  
        if isinstance(self.learning_rate, float):
            self.lr = lambda n: float(self.learning_rate)
        else: 
            self.lr = self.learning_rate
        
    def __str__(self):
        return "MyLineReg class: " + ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
    
    def _add_const(self, X):
        _X = X.copy()
        _X.insert(0, 'const', 1, allow_duplicates=True)
        return _X
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        self._X = self._add_const(X).to_numpy()
        self._y = y.to_numpy()
        self.weights = np.ones(self._X.shape[1])
        for i in range(1, self.n_iter + 1):
            y_pred = self._X @ self.weights
            loss = self.mse(self._y, y_pred)
            grad = 2 * (y_pred - self._y) @ self._X / len(self._X)
            if self.reg == "l1":
                loss += self.l1_coef * np.sum(np.abs(self.weights))
                grad += self.l1_coef * np.sign(self.weights)
            elif self.reg == "l2":
                loss += self.l2_coef * np.sum(self.weights**2)
                grad += self.l2_coef * 2 * self.weights
            elif self.reg == "elasticnet":
                loss += self.l1_coef * np.sum(np.abs(self.weights)) * self.l2_coef * np.sum(self.weights**2)
                grad += self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights
            self.weights -= self.lr(i) * grad
            if verbose and i % verbose == 0:
                if self.metrics:
                    print(f"{i} | loss: {loss} |")
                else:
                    print(f"{i} | loss: {loss} | metrics")
                
    def get_coef(self):
        return self.weights[1:]
    
    def predict(self, X):
        _X = self._add_const(X).to_numpy()
        y_pred = _X @ self.weights
        return np.sum(y_pred)
    
    def mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    def rmse(self, y_true, y_pred):
        return np.sqrt(self.mse(y_true, y_pred))
    
    def mape(self, y_true, y_pred):
        return 100 * np.mean(np.abs((y_true - y_pred) / y_true))
    
    def r2(self, y_true, y_pred):
        return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    
    def get_best_score(self):
        return getattr(self, self.metric)(self._y, self._X @ self.weights)

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

reg = MyLineReg(**{"n_iter": 50, "learning_rate": 0.1, "metric": 'mape'})
reg.fit(X, y, 10)
print(np.sum(reg.get_coef()))
print(reg.get_best_score())