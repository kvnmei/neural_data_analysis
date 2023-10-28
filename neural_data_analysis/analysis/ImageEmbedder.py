#!/usr/bin/env python3
"""
This module contains functions to embed images using different models.

Functions:
    None

Classes:


Examples:

"""

from typing import Protocol, List, Dict

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoProcessor,
)

embedder_config = {
    "ResNet50Embedder": {
        "embedding_name": "resnet",
        "batch_size": 64,
    },
    "CLIPEmbedder": {
        "embedding_name": "clip",
        "batch_size": 512,
        "model": "openai/clip-vit-base-patch32",
        "processor": "openai/clip-vit-base-patch32",
    },
    "rgbhsvl": {"embedding_name": "rgbhsvl",},
    "gist": {"embedding_name": "gist",},
    "moten": {"embedding_name": "moten",},
    # "blip_encoder": {
    #     "name": "blip",
    #     "batch_size": "64",
    #     "model": "Salesforce/blip-image-captioning-base",
    #     "processor": "Salesforce/blip-image-captioning-base",
    # },
    # "object_detection": {
    #     "name": "detr",
    #     "model": "facebook/detr-resnet-50",
    #     "processor": "facebook/detr-resnet-50",
    #     "detection_threshold": "0.5",
    # },
}


class ImageEmbedder(Protocol):
    """
    Embeds input images by an encoding model.

    Attributes:
        config (dict): configuration dictionary
        device (torch.device): device to use for embedding

    Methods:
        preprocess(self, encoder)
        embed(self, encoder)

    """

    config: dict
    device: torch.device
    batch_size: int

    def preprocess(self, images: torch.Tensor) -> dict:
        pass

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        pass


class ResNet50Embedder(ImageEmbedder, nn.Module):
    def __init__(self, config: dict, device: torch.device) -> None:
        super().__init__()
        nn.Module.__init__(self)
        self.config = config
        self.device = device
        self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.encoder.fc = nn.Identity()
        self.encoder.to(device)
        self.encoder.eval()
        self.processor = ResNet50_Weights.IMAGENET1K_V2.transforms()
        self.batch_size = embedder_config["ResNet50Embedder"]["batch_size"]

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        # Tensor should be (n_samples, n_channels, height, width)
        if 'numpy' in str(type(images)):
            images = torch.from_numpy(images)
        try:
            assert images.shape[1] == 3
        except AssertionError:
            print("Images must be in (n_samples, n_channels, height, width) format.")
            images = images.permute(0, 3, 1, 2)

        results = []
        print("Embedding with ResNet...")
        for i in tqdm(range(0, images.shape[0], self.batch_size)):
            batch = images[i : i + self.batch_size]
            batch = self.processor(batch).to(self.device)
            batch = self.encoder(batch)
            results.append(batch)
        results = torch.cat(results, dim=0)
        return results


class CLIPEmbedder(ImageEmbedder, nn.Module):
    def __init__(self, config: dict, device: torch.device) -> None:
        super().__init__()
        nn.Module.__init__(self)
        self.config = config
        self.device = device
        self.encoder = CLIPModel.from_pretrained(embedder_config["CLIPEmbedder"]["model"])
        self.encoder.to(device)
        self.encoder.eval()
        self.processor = AutoProcessor.from_pretrained(embedder_config["CLIPEmbedder"]["processor"])
        self.batch_size = embedder_config["CLIPEmbedder"]["batch_size"]

    def preprocess(self, images: torch.Tensor) -> dict:
        if 'numpy' in str(type(images)):
            images = torch.from_numpy(images)
        # to PIL image
        images = images.permute(0, 3, 1, 2)
        images = [torchvision.transforms.ToPILImage()(img) for img in images]
        images = self.processor(
            # TODO: change the text from "a" to ""
            # text=["a"] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        return images

    # noinspection PyPep8
    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        results = []
        print("Embedding with CLIP...")
        for i in tqdm(range(0, images.shape[0], self.batch_size)):
            batch = images[i : i + self.batch_size]
            batch = self.preprocess(batch)  # returns a dict-like object
            batch = self.encoder.get_image_features(**batch)
            results.append(batch)
            # results.append(batch["image_embeds"])
        results = torch.cat(results, dim=0)
        return results


# TODO: object detection
# def __init__(self, config: dict, device: torch.device, model: str):
#     super(ImageEmbedder, self).__init__()
#
#     self.config = config
#     self.device = device
#     self.model = model
#
#     self.encoder = None
#     self.processor = None
#     self.batch_size = None
# self.obj_detection_threshold = None
# self.text_prompt = None
# def init_encoder(self):
# elif self.model == "blip":
#     self.encoder = BlipForConditionalGeneration.from_pretrained(
#         self.config["imageEmbedder"]["blip_encoder"]["model"]
#     ).to(self.device)
#     self.processor = AutoProcessor.from_pretrained(
#         self.config["imageEmbedder"]["blip_encoder"]["processor"]
#     )
#     self.text_prompt = "A picture of "
#     self.batch_size = self.config["imageEmbedder"]["blip_encoder"]["batch_size"]
# elif self.model == "object_detection":
#     self.encoder = AutoModelForObjectDetection.from_pretrained(
#         self.config["imageEmbedder"]["object_detection"]["processor"]
#     )
#     self.processor = AutoImageProcessor.from_pretrained(
#         self.config["imageEmbedder"]["object_detection"]["processor"]
#     )
#     self.obj_detection_threshold = self.config["imageEmbedder"][
#         "object_detection"
#     ]["detection_threshold"]
# elif self.model == "vit":
# TODO: finish implementing ViT
# self.encoder = vit_b_16(weights=ViT_B_16_Weights)
# self.encoder.to(self.device)
# self.processor = ViT_B_16_Weights.transforms
# self.encoder.eval()
# self.batch_size = self.config["imageEmbedder"]["vit_encoder"]["batch_size"]
# else:
#     print(f"Model {self.model} not implemented for imageEmbedder class.")
# @torch.no_grad()
# def get_blip_embeddings(self, video: torch.Tensor):
#     result = []
#     result_batch = []
#     with torch.no_grad():
#         for i in tqdm(range(0, len(video), self.batch_size)):
#             batch = video[i : i + self.batch_size]
#             batch = batch.to(self.device)
#             batch = self.processor(
#                 text=[self.text_prompt] * len(batch),
#                 images=[batch[j] for j in range(batch.shape[0])],
#                 return_tensors="pt",
#                 padding=True,
#             ).to(self.device)
#             batch = self.encoder.generate(**batch)
#             r = self.processor.batch_decode(batch, skip_special_tokens=True)
#             result.append(r)
#             result_batch.append(batch)
#     # flatten the list
#     result = [item for sublist in result for item in sublist]
#     result_batch = [item for sublist in result_batch for item in sublist]
#     return result, result_batch

# @torch.no_grad()
# def get_object_detection_embeddings(self, images: torch.Tensor):
#     target_sizes = [[img.shape[0], img.shape[1]] for img in images]
#     try:
#         target_sizes = [
#             torch.tensor([img.shape[0], img.shape[1]]) for img in images
#         ]
#     except ValueError:
#         print("Video must be a list of tensors.")
#
#     images = images.permute(0, 3, 1, 2)
#     images = [transforms.ToPILImage()(img) for img in images]
#
#     image = self.processor(images, return_tensors="pt")
#     image = self.encoder(**image)
#
#     result = self.processor.post_process_object_detection(
#         image, threshold=self.obj_detection_threshold, target_sizes=target_sizes
#     )
#     return result

# def plot_object_detection_result(self, result, ax=None):
#     # indicate the bounding box
#     for box, label in zip(result["boxes"], result["labels"]):
#         box = box.detach().numpy()
#         # decode label
#         label = self.encoder.config.id2label[label.item()]
#         ax.plot(
#             [box[0], box[2], box[2], box[0], box[0]],
#             [box[1], box[1], box[3], box[3], box[1]],
#             label=label,
#         )
#     ax.legend()
#
#     return ax
# elif embedding_name == "blip":
#     words, embedding = image_embedder.get_blip_embeddings(image)
#     print(words)
# elif embedding_name == "object_detection":
#     obj_result = image_embedder.get_object_detection_embeddings(image)
#     fig, ax = plt.subplots()
#     ax.imshow(image[0])
#     image_embedder.plot_object_detection_result(obj_result[0], ax=ax)
#     plt.show()


def embedder_from_spec(embedder_name: str) -> ImageEmbedder:
    """
    Create an embedder from a specification dictionary.

    Args:
        embedder_name (str): name of Embedder to create

    Returns:
        embedder (ImageEmbedder): model to embed images
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if embedder_name == "ResNet50Embedder":
        spec = embedder_config[embedder_name]
        return ResNet50Embedder(spec, device)
    elif embedder_name == "CLIPEmbedder":
        spec = embedder_config[embedder_name]
        return CLIPEmbedder(spec, device)
    else:
        raise NotImplementedError(f"Embedder {embedder_name} not implemented.")


# NOT USED
def create_image_embeddings(
    images: torch.Tensor,
    embedder_list: List[str],
    embedder_specs: dict,
) -> Dict[str, torch.Tensor]:
    """
    Create embeddings for a list of images using a list of embedders.

    Args:
        images (torch.Tensor): images to embed
        embedder_list (list): list of embedder names
        embedder_specs (dict): dictionary of embedder specifications

    Returns:
        embeddings (dict): dictionary of embeddings
    """
    embeddings = {}
    for embedder_name in embedder_list:
        print(f"Model {embedder_name}")
        image_embedder = embedder_from_spec(embedder_specs[embedder_name])
        embedding = image_embedder.embed(images)
        embeddings[embedder_specs[embedder_name]["embedding_name"]] = embedding
    return embeddings
