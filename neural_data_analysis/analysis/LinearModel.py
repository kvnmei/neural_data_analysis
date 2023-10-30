"""

"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
#TODO: instantiate a check that only a single float variable is the target like
# if (self.config["model"]["linear_regression"] == "pls") & (
# self.config["model"]["targets"] == "single"
#            ):

def ols_solver(x, y):
    betas = (
        np.linalg.inv(x.T @ x) @ x.T @ y
    )  # the @ symbol is equivalent to np.matmul()
    return betas


def ridge_solver(x, y, alpha):
    betas = np.linalg.inv(x.T @ x + alpha * np.identity(x.shape[1])) @ x.T @ y
    return betas

class LinearModel:
    def __init__(self, config):
        self.config = config

    def PLSR(self):
        plsr = PLSRegression(n_components=10, scale=True)
        _ = plsr.fit(X_train, y_train)
        preds = plsr.predict(X_val).flatten()
