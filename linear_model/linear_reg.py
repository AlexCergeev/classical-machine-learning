import pandas as pd
import numpy as np

from sklearn.datasets import make_regression

class MyLineReg():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in self.kwargs.items():
            setattr(self, key, value)
        
    def __str__(self):
        return "MyLineReg class: " + ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
    
    def _add_const(self, X):
        _X = X.copy()
        _X.insert(0, 'const', 1, allow_duplicates=True)
        return _X
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        _X = self._add_const(X).to_numpy()
        _y = y.to_numpy()
        self.weights = np.ones(_X.shape[1])
        for i in range(1, self.kwargs["n_iter"] + 1):
            y_ped = _X @ self.weights
            mse = np.mean((_y - y_ped)**2)
            grad = 2 * (y_ped - _y) @ _X / len(_X)
            self.weights -= self.kwargs["learning_rate"] * grad
            if verbose and i % verbose == 0:
                print(f"{i} | {mse}")
                
    def get_coef(self):
        return self.weights[1:]

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

reg = MyLineReg(**{"n_iter": 50, "learning_rate": 0.1})
reg.fit(X, y, 10)
print(reg.get_coef())