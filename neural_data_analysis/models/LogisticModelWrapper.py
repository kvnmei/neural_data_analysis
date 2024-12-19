from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression


class LogisticModelWrapper:
    def __init__(self, config: dict):
        self.config = config
        if self.config["problem_type"] == "multilabel_binary_classification":
            # look into binary classification, tweaking params: solver, max_iter, tol, C
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        MultiOutputClassifier(
                            LogisticRegression(
                                class_weight=self.config["class_weight"],
                                solver=self.config["solver"],
                                max_iter=self.config["max_iter"],
                            )
                        ),
                    ),
                ]
            )
            self.model = pipe

    def fit(self, X, y):
        if self.config["backend"] == "sklearn":
            self.model.fit(X, y)

    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions
