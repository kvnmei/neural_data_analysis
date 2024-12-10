"""
"""
import numpy as np
import yaml
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# TODO: instantiate a check that only a single float variable is the target like
# if (self.config["model"]["linear_regression"] == "pls") & (
# self.config["model"]["targets"] == "single"
#            ):


# noinspection PyPep8Naming,PyShadowingNames
class LinearModelWrapper:
    """Wrapper for linear models."""

    def __init__(self, config: dict, method: str = "ols", alpha: float = 0.05):
        """
        Initialize the LinearModel class.

        Args:
            config (dict):
            method (str):
            alpha (float):
        """
        self.config = config
        self.hparams = {
            "problem_type": config["problem_type"],
            "backend": config["backend"],
        }
        self.method = method
        self.alpha = alpha
        self.coef_ = None
        if self.config["problem_type"] == "binary_classification":
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(class_weight="balanced", max_iter=1000),
                    ),
                ]
            )
            self.model = pipeline
        elif self.config["problem_type"] == "multilabel_binary_classification":
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        MultiOutputClassifier(
                            LogisticRegression(class_weight="balanced", max_iter=1000)
                        ),
                    ),
                ]
            )
            self.model = pipeline
        print("Logistic Regression uses balanced class weights.")

    # noinspection PyShadowingNames
    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.

        Args:
            X (numpy.ndarray): Independent variables (features) with shape (n_samples, n_features).
            y (numpy.ndarray): Dependent variable with shape (n_samples,).

        Returns:
            self (LinearModel): Returns an instance of the LinearModel class.
        """

        if self.config["backend"] == "custom":
            if self.method == "ols":
                self.coef_ = self.ols_regression(X, y)
            elif self.method == "ridge":
                self.coef_ = self.ridge_regression(X, y, self.alpha)
            elif self.method == "plsr":
                self.pls_regression(X, y)
            else:
                raise ValueError(
                    "Invalid regression method. Choose from 'ols', 'ridge', or 'plsr'."
                )
        elif self.config["backend"] == "sklearn":
            self.model.fit(X, y)

    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions

    @staticmethod
    def ridge_regression(X, y, alpha):
        """
        Solve Ridge regression.

        Args:
        - X (numpy.ndarray): Independent variables (features) with shape (n_samples, n_features).
        - y (numpy.ndarray): Dependent variable with shape (n_samples,).
        - alpha (float): Regularization parameter.

        Returns:
        - betas (numpy.ndarray): Coefficients of the Ridge regression model.
        """
        betas = np.linalg.inv(X.T @ X + alpha * np.identity(X.shape[1])) @ X.T @ y
        return betas

    @staticmethod
    def ols_regression(X, y):
        """
        Solve Ordinary Least Squares regression.

        Args:
        - X (numpy.ndarray): Independent variables (features) with shape (n_samples, n_features).
        - y (numpy.ndarray): Dependent variable with shape (n_samples,).

        Returns:
        - betas (numpy.ndarray): Coefficients of the linear regression model.
        """
        betas = np.linalg.inv(X.T @ X) @ X.T @ y
        return betas

    def pls_regression(self, X, y):
        """
        Fit Partial Least Squares Regression model.

        Args:
            X (numpy.ndarray): Independent variables (features) with shape (n_samples, n_features).
            y (numpy.ndarray): Dependent variable with shape (n_samples,).
        """
        plsr = PLSRegression(n_components=2, scale=True)
        _ = plsr.fit(X, y)
        self.coef_ = plsr.coef_
        return plsr.coef_

    # def log_regression(self, X, y):


if __name__ == "__main__":
    cfg = yaml.load(open("../analysis/tests/config.yaml", "r"), Loader=yaml.FullLoader)

    # Create some random data
    np.random.seed(0)
    X = 2 * np.random.rand(100, 5)  # 10 features
    y = 4 + np.dot(X, np.array([50, 40, 30, 20, 10])) + np.random.randn(100)

    # Instantiate LinearModel with OLS method
    model_ols = LinearModelWrapper(config=cfg, method="ols")

    # Fit the model to the data
    model_ols.fit(X, y)

    # Print the coefficients
    print("Coefficients (OLS):", model_ols.coef_)

    # Instantiate LinearModel with Ridge method
    model_ridge = LinearModelWrapper(config=cfg, method="ridge", alpha=0.1)

    # Fit the model to the data
    model_ridge.fit(X, y)

    # Print the coefficients
    print("Coefficients (Ridge):", model_ridge.coef_)

    # Instantiate LinearModel with PLSR method
    model_plsr = LinearModelWrapper(config=cfg, method="plsr")

    # Fit the model to the data
    model_plsr.fit(X, y)

    # PLSR doesn't return coefficients directly, but you can access other properties like model.coef_
    print("Coefficients (PLSR):", model_plsr.coef_)

    print("Done")
