import unittest
import numpy as np
import yaml

from neural_data_analysis import LogisticModelWrapper


class TestLogisticProcessor(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        self.y_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 1]])
        self.X_test = np.array([[2, 3], [4, 5]])
        self.y_test = np.array([[1, 0], [0, 0]])

        with open("config.yaml", "r") as cfg:
            self.config = yaml.load(cfg, Loader=yaml.FullLoader)

        self.model = LogisticModelWrapper(self.config)

    def
