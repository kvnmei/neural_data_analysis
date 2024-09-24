import unittest

import numpy as np
import requests
import torch
import yaml
from PIL import Image

from neural_data_analysis import embedder_configs
from neural_data_analysis.analysis import create_image_embeddings, embedder_from_spec


class TestRandomData(unittest.TestCase):
    def setUp(self):
        with open("config.yaml", "r") as cfg:
            self.config = yaml.load(cfg, Loader=yaml.FullLoader)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.rand(1, 224, 224, 3)

    def test_embed(self):
        embeddings = create_image_embeddings(
            self.data,
            self.config["ImageLoader"]["embedders_to_use"],
            self.config["ImageEmbedder"],
        )
        assert len(embeddings) == len(self.config["ImageLoader"]["embedders_to_use"])

    def test_DETREmbedder(self):
        embedder_name = "DETREmbedder"
        image_embedder = embedder_from_spec(embedder_name, embedder_configs)
        embedding = image_embedder.embed(self.data)


class TestRealImage(unittest.TestCase):
    def setUp(self):
        with open("config.yaml", "r") as cfg:
            self.config = yaml.load(cfg, Loader=yaml.FullLoader)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        url = "https://t3.ftcdn.net/jpg/03/28/26/86/360_F_328268657_OSSRDoZeO6isXj3QxRUTaXlmGF6lLmyZ.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        image = image.resize((224, 224))
        image = np.array(image)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        self.data = image

    def test_embed(self):
        embeddings = create_image_embeddings(
            self.data,
            self.config["ImageLoader"]["embedders_to_use"],
            self.config["ImageEmbedder"],
        )
        assert len(embeddings) == len(self.config["ImageLoader"]["embedders_to_use"])

    def test_DETREmbedder(self):
        embedder_name = "DETREmbedder"
        image_embedder = embedder_from_spec(embedder_name, embedder_configs)
        object_bounding_boxes = image_embedder.embed(self.data)

    def test_plot_object_detection_result(self, result, ax=None):
        pass


if __name__ == "__main__":
    unittest.main()
