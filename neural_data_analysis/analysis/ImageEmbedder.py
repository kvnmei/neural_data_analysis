#!/usr/bin/env python3
"""
This module contains functions to embed images using different models.

Functions:
    None

Classes:


Examples:

"""
from typing import Dict

import numpy as np
import requests
import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    AutoProcessor,
    BlipForConditionalGeneration,
    CLIPModel,
    CLIPProcessor,
)


class ImageEmbedder(nn.Module):
    """
    You can pass an input into the class that will be embedding by the encoding model.

    Attributes:
        config (dict): configuration dictionary
        device (torch.device): device to use for embedding
        encoder (model): model to use for embedding

    Methods:
        preprocess(self, encoder)
        get_resnet_embeddings(self, input)
        get_clip_embeddings(self, input)
        forward(self, input, encoder)

    Examples:
    """

    def __init__(self, config: Dict, device: torch.device, model: str):
        super(ImageEmbedder, self).__init__()

        self.config = config
        self.device = device
        self.model = model

        self.encoder = None
        self.processor = None
        self.batch_size = None
        self.obj_detection_threshold = None
        self.text_prompt = None
        self.init_encoder()

    def init_encoder(self):
        if self.model == "resnet":
            self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.encoder.fc = nn.Identity()
            self.encoder.to(self.device)
            self.encoder.eval()
            self.processor = ResNet50_Weights.IMAGENET1K_V2.transforms()
            self.batch_size = self.config["imageEmbedder"]["resnet_encoder"][
                "batch_size"
            ]

        elif self.model == "clip":
            self.encoder = CLIPModel.from_pretrained(
                self.config["imageEmbedder"]["clip_encoder"]["model"]
            )
            self.encoder.to(self.device)
            self.encoder.eval()
            self.processor = CLIPProcessor.from_pretrained(
                self.config["imageEmbedder"]["clip_encoder"]["processor"]
            )
            self.batch_size = self.config["imageEmbedder"]["clip_encoder"]["batch_size"]
        elif self.model == "blip":
            self.encoder = BlipForConditionalGeneration.from_pretrained(
                self.config["imageEmbedder"]["blip_encoder"]["model"]
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                self.config["imageEmbedder"]["blip_encoder"]["processor"]
            )
            self.text_prompt = "A picture of "
            self.batch_size = self.config["imageEmbedder"]["blip_encoder"]["batch_size"]
        elif self.model == "object_detection":
            self.encoder = AutoModelForObjectDetection.from_pretrained(
                self.config["imageEmbedder"]["object_detection"]["processor"]
            )
            self.processor = AutoImageProcessor.from_pretrained(
                self.config["imageEmbedder"]["object_detection"]["processor"]
            )
            self.obj_detection_threshold = self.config["imageEmbedder"][
                "object_detection"
            ]["detection_threshold"]
        elif self.model == "vit":
            # TODO: finish implementing ViT
            self.encoder = vit_b_16(weights=ViT_B_16_Weights)
            self.encoder.to(self.device)
            self.processor = ViT_B_16_Weights.transforms
            self.encoder.eval()
            self.batch_size = self.config["imageEmbedder"]["vit_encoder"]["batch_size"]
        else:
            print(f"Model {self.model} not implemented for imageEmbedder class.")

    def preprocess(self, x: torch.Tensor):
        if self.model == "resnet":
            x = self.processor(x)
        elif self.model == "clip":
            x = self.processor(
                text=["a"] * len(x),
                images=[x[i] for i in range(x.shape[0])],
                return_tensors="pt",
                padding=True,
            ).to(self.device)
        else:
            print(f"No preprocessing for {self.model} model.")
        return x

    @torch.no_grad()
    def get_resnet_embeddings(self, x: torch.Tensor):
        # Tensor should be (n_samples, n_channels, height, width)
        results = []
        print("Embedding with ResNet...")
        for i in tqdm(range(0, x.shape[0], self.batch_size)):
            batch = x[i : i + self.batch_size]
            batch = self.preprocess(batch)
            batch = self.encoder(batch)
            results.append(batch)
        results = torch.cat(results, dim=0)
        return results

    @torch.no_grad()
    def get_clip_embeddings(self, x: torch.Tensor):
        results = []
        print("Embedding with CLIP...")
        for i in tqdm(range(0, x.shape[0], self.batch_size)):
            batch = x[i : i + self.batch_size]
            batch = self.preprocess(batch)
            batch = self.encoder(**batch)
            results.append(batch["image_embeds"])
        results = torch.cat(results, dim=0)
        return results

    @torch.no_grad()
    def get_blip_embeddings(self, video: torch.Tensor):
        result = []
        result_batch = []
        with torch.no_grad():
            for i in tqdm(range(0, len(video), self.batch_size)):
                batch = video[i : i + self.batch_size]
                batch = batch.to(self.device)
                batch = self.processor(
                    text=[self.text_prompt] * len(batch),
                    images=[batch[j] for j in range(batch.shape[0])],
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                batch = self.encoder.generate(**batch)
                r = self.processor.batch_decode(batch, skip_special_tokens=True)
                result.append(r)
                result_batch.append(batch)
        # flatten the list
        result = [item for sublist in result for item in sublist]
        result_batch = [item for sublist in result_batch for item in sublist]
        return result, result_batch

    @torch.no_grad()
    def get_object_detection_embeddings(self, images: torch.Tensor):
        target_sizes = [[img.shape[0], img.shape[1]] for img in images]
        try:
            target_sizes = [
                torch.tensor([img.shape[0], img.shape[1]]) for img in images
            ]
        except ValueError:
            print("Video must be a list of tensors.")

        images = images.permute(0, 3, 1, 2)
        images = [transforms.ToPILImage()(img) for img in images]

        image = self.processor(images, return_tensors="pt")
        image = self.encoder(**image)

        result = self.processor.post_process_object_detection(
            image, threshold=self.obj_detection_threshold, target_sizes=target_sizes
        )
        return result

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        if self.model == "resnet":
            return self.get_resnet_embeddings(x)
        elif self.model == "clip":
            return self.get_clip_embeddings(x)
        else:
            print(f"No forward for {self.model} model.")

    def plot_object_detection_result(self, result, ax=None):
        # indicate the bounding box
        for box, label in zip(result["boxes"], result["labels"]):
            box = box.detach().numpy()
            # decode label
            label = self.encoder.config.id2label[label.item()]
            ax.plot(
                [box[0], box[2], box[2], box[0], box[0]],
                [box[1], box[1], box[3], box[3], box[1]],
                label=label,
            )
        ax.legend()

        return ax


def random_tests(config, device):
    x = torch.rand(1, 224, 224, 3)
    for embedding_name in config["imageLoader"]["embeddings_to_use"]:
        print(f"Model {embedding_name}")
        image_embedder = ImageEmbedder(config, device, embedding_name)
        if embedding_name == "resnet":
            result = image_embedder.forward(x)
            print(result.shape)
        elif embedding_name == "clip":
            result = image_embedder.forward(x)
            print(result.shape)
        elif embedding_name == "blip":
            words, embedding = image_embedder.get_blip_embeddings(x)
            print(words)
        elif embedding_name == "object_detection":
            result = image_embedder.get_object_detection_embeddings(x)
            _ = plt.figure()
            image_embedder.plot_object_detection_result(result[0])
            plt.show()


def real_image_test(config, device):
    url = "https://t3.ftcdn.net/jpg/03/28/26/86/360_F_328268657_OSSRDoZeO6isXj3QxRUTaXlmGF6lLmyZ.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image = image.resize((224, 224))
    image = np.array(image)
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    for embedding_name in config["imageLoader"]["embeddings_to_use"]:
        print(f"Model {embedding_name}")
        image_embedder = ImageEmbedder(config, device, embedding_name)
        if embedding_name == "resnet":
            result = image_embedder.forward(image)
            print(result.shape)
        elif embedding_name == "clip":
            result = image_embedder.forward(image)
            print(result.shape)
        elif embedding_name == "blip":
            words, embedding = image_embedder.get_blip_embeddings(image)
            print(words)
        elif embedding_name == "object_detection":
            obj_result = image_embedder.get_object_detection_embeddings(image)
            fig, ax = plt.subplots()
            ax.imshow(image[0])
            image_embedder.plot_object_detection_result(obj_result[0], ax=ax)
            plt.show()


if __name__ == "__main__":
    cfg = yaml.load(open("../config.yaml", "r"), Loader=yaml.FullLoader)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_tests(cfg, dev)
    real_image_test(cfg, dev)
