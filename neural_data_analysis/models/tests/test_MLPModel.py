import unittest

import torch
import yaml
from sklearn.metrics import r2_score

from neural_data_analysis.analysis import MLPModelWrapper


class TestRandomData(unittest.TestCase):
    """
    Test the MLPModel class on random data.
    """

    def setUp(self):
        with open("../../analysis/tests/config.yaml", "r") as cfg:
            self.config = yaml.load(cfg, Loader=yaml.FullLoader)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_fit(self):
        X_train = torch.rand(10000, 100)
        y_train = torch.rand(10000, 100)
        X_val = torch.rand(1000, 100)
        y_val = torch.rand(1000, 100)
        model = MLPModelWrapper(self.config, X_train.shape[-1], y_train.shape[-1])
        model.fit(X_train, y_train, X_val, y_val)
        y_test = torch.rand(100, 100)
        predictions = model.predict(y_test)
        gt = torch.rand(100, 100)
        score = r2_score(gt, predictions)
        print(score)


if __name__ == "__main__":
    unittest.main()
