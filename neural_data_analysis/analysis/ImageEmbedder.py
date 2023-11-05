#!/usr/bin/env python3
"""
This module contains functions to embed images using different models.

Functions:
    None

Classes:


Examples:

"""

from typing import Dict, List, Protocol

import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from moviepy.video.io.VideoFileClip import VideoFileClip
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
from transformers import (AutoImageProcessor, AutoModelForObjectDetection, AutoProcessor, Blip2ForConditionalGeneration,
                          Blip2Processor, BlipForConditionalGeneration, CLIPModel)

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
    "DETREmbedder": {
        "embedding_name": "detr",
        "embedding_description": "DEtection TRansformer for object detection",
        "model": "facebook/detr-resnet-50",
        "processor": "facebook/detr-resnet-50",
        "detection_threshold": 0.5,
        "batch_size": 16,
    },
    "BLIPEmbedder": {
        "embedding_name": "blip",
        "embedding_description": "BLIP: Learning Better Language Models by Encoding Images in Text Sequences",
        "batch_size": 64,
        "model": "Salesforce/blip-image-captioning-base",
        "processor": "Salesforce/blip-image-captioning-base",
    },
    "BLIP2Embedder": {
        "embedding_name": "blip2",
        "embedding_description": "BLIP: Learning Better Language Models by Encoding Images in Text Sequences",
        "batch_size": 64,
        "model": "Salesforce/blip2-opt-2.7b",
        "processor": "Salesforce/blip2-opt-2.7b",
    },
    "ViT_B_16Embedder": {
        "embedding_name": "vit",
        "embedding_description": "Vision Transformer Base 16",
        "batch_size": 64,
    },
    "rgbhsvl": {"embedding_name": "rgbhsvl", },
    "gist": {"embedding_name": "gist", },
    "moten": {"embedding_name": "moten", },
    "face_regressors": {"embedding_name": "face_regressors", },
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

    def __init__(self):
        self.encoder = None

    def preprocess(self, images: torch.Tensor) -> dict:
        pass

    def embed(self, images: torch.Tensor):
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
        self.batch_size = config["batch_size"]

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> np.ndarray:
        """

        Args:
            images (torch.Tensor): (n_samples, n_channels, height, width)
        """
        images = _check_image_tensor_dimensions(images)

        results = []
        print("Embedding with ResNet...")
        for i in tqdm(range(0, images.shape[0], self.batch_size)):
            batch = images[i: i + self.batch_size]
            batch = self.processor(batch).to(self.device)
            batch = self.encoder(batch)
            results.append(batch)
        results = torch.cat(results, dim=0).cpu().numpy()
        return results


class CLIPEmbedder(ImageEmbedder, nn.Module):
    def __init__(self, config: dict, device: torch.device) -> None:
        super().__init__()
        nn.Module.__init__(self)
        self.config = config
        self.device = device
        self.encoder = CLIPModel.from_pretrained(config["model"])
        self.encoder.to(device)
        self.encoder.eval()
        self.processor = AutoProcessor.from_pretrained(config["processor"])
        self.batch_size = config["batch_size"]

    def preprocess(self, images: torch.Tensor) -> dict:
        if 'numpy' in str(type(images)):
            images = torch.from_numpy(images)
        # to PIL image
        images = _check_image_tensor_dimensions(images)
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
    def embed(self, images: torch.Tensor) -> np.ndarray:
        results = []
        print("Embedding with CLIP...")
        for i in tqdm(range(0, images.shape[0], self.batch_size)):
            batch = images[i: i + self.batch_size]
            batch = self.preprocess(batch)  # returns a dict-like object
            batch = self.encoder.get_image_features(**batch)
            results.append(batch)
            # results.append(batch["image_embeds"])
        results = torch.cat(results, dim=0).cpu().numpy()
        return results


class DETREmbedder(ImageEmbedder):
    def __init__(self, config: dict, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.batch_size = config["batch_size"]
        self.obj_detection_threshold = config["detection_threshold"]
        self.encoder = AutoModelForObjectDetection.from_pretrained(
            config["processor"]
        )
        self.encoder.to(device)
        self.processor = AutoImageProcessor.from_pretrained(
            config["processor"]
        )

    def preprocess(self, images):
        images = self.processor(images, return_tensors="pt").to(self.device)
        return images

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> List[dict]:
        """
        Args:
            images (torch.Tensor): (n_samples, n_channels, height, width)

        """
        images = _check_image_tensor_dimensions(images)

        # target sizes should be (height, width) and used to rescale the object detection predictions to the original
        # image size; passing in images as (n_channels, height, width) so we use .shape[1] and .shape[2] to get the
        # height and width
        target_sizes = [[img.shape[1], img.shape[2]] for img in images]
        try:
            target_sizes = [
                torch.tensor([img.shape[1], img.shape[2]]) for img in images
            ]
        except ValueError:
            print("Video must be a list of tensors.")

        # PIL images are (width, height)
        images = [transforms.ToPILImage()(img) for img in images]

        result_obj_ids = []
        result_obj_names = []
        result_obj_boxes = []
        result_obj_scores = []
        print("Embedding with DETR...")
        for i in tqdm(range(0, len(images), self.batch_size)):
            batch = images[i: i + self.batch_size]
            batch_target_sizes = target_sizes[i: i + self.batch_size]
            inputs = self.preprocess(batch)
            outputs = self.encoder(**inputs)
            result = self.processor.post_process_object_detection(
                outputs,
                threshold=self.obj_detection_threshold,
                target_sizes=batch_target_sizes
            )
            result_ids = [res["labels"] for res in result]
            result_names = [[self.encoder.config.id2label[id.item()] for id in ids] for ids in result_ids]
            result_scores = [res["scores"] for res in result]
            result_boxes = [res["boxes"] for res in result]
            result_obj_ids.extend(result_ids)
            result_obj_names.extend(result_names)
            result_obj_scores.extend(result_scores)
            result_obj_boxes.extend(result_boxes)
        result_obj_ids = [ids.cpu().numpy() for ids in result_obj_ids]
        result_obj_names = [np.array(names) for names in result_obj_names]
        result_obj_boxes = [boxes.cpu().numpy() for boxes in result_obj_boxes]
        result_obj_scores = [scores.cpu().numpy() for scores in result_obj_scores]
        results = {
            "ids": result_obj_ids,
            "labels": result_obj_names,
            "boxes": result_obj_boxes,
            "scores": result_obj_scores,
        }
        return results


class BLIPEmbedder(ImageEmbedder):
    def __init__(self, config: dict, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.encoder = BlipForConditionalGeneration.from_pretrained(
            self.config["model"]
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.config["processor"]
        )
        self.text_prompt = "A picture of "
        self.batch_size = self.config["batch_size"]

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        result_text = []
        result_ids = []
        print("Embedding with BLIP...")
        with torch.no_grad():
            for i in tqdm(range(0, len(images), self.batch_size)):
                batch = images[i: i + self.batch_size]
                batch = batch.to(self.device)
                batch = self.processor(
                    text=[self.text_prompt] * len(batch),
                    images=[batch[j] for j in range(batch.shape[0])],
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                ids = self.encoder.generate(**batch)
                text = self.processor.batch_decode(batch, skip_special_tokens=True)
                result_text.append(text)
                result_ids.append(batch)
        # flatten the list
        result_text = [item for sublist in result_text for item in sublist]
        result_ids = [item.cpu().numpy() for sublist in result_ids for item in sublist]
        results = {
            "embeddings": result_ids,
            "captions": result_text,
        }
        return results


class BLIP2Embedder(ImageEmbedder):
    def __init__(self, config: dict, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.processor = Blip2Processor.from_pretrained(
            self.config["processor"]
        )
        self.encoder = Blip2ForConditionalGeneration.from_pretrained(
            self.config["model"],
            torch_dtype=torch.float16,
        ).to(self.device)

        # self.text_prompt = "A picture of "
        self.batch_size = self.config["batch_size"]

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        result_text = []
        result_ids = []
        image_ids = []
        current_image_id = 0
        print("Embedding with BLIP2...")
        with torch.no_grad():
            for i in tqdm(range(0, len(images), self.batch_size)):
                batch = images[i: i + self.batch_size]
                batch = batch.to(self.device)
                batch = self.processor(
                    # text=[self.text_prompt] * len(batch),
                    images=batch,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                ids = self.encoder.generate(**batch)
                text = self.processor.batch_decode(ids, skip_special_tokens=True)
                result_ids.append(ids)
                result_text.append(text)
                for el in range(len(text)):
                    # print(f"Image {current_image_id}: {text[el]}")
                    image_ids.append(current_image_id)
                    current_image_id += 1
        # flatten the list
        result_ids = [item.cpu().numpy() for sublist in result_ids for item in sublist]
        result_text = [item for sublist in result_text for item in sublist]
        results = {
            "ids": result_ids,
            "text": result_text,
            "image_ids": image_ids,
        }
        return results


class ViTEmbedder(ImageEmbedder):
    # TODO: finish implementing ViT

    def __init__(self, config: dict, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.encoder = vit_b_16(weights=ViT_B_16_Weights)
        self.encoder.to(self.device)
        self.processor = ViT_B_16_Weights.transforms
        self.encoder.to(device)
        self.encoder.eval()
        self.batch_size = self.config["batch_size"]


def embedder_from_spec(embedder_name: str, device: str = None) -> ImageEmbedder:
    """
    Create an embedder from a specification dictionary.

    Args:
        embedder_name (str): name of Embedder to create

    Returns:
        embedder (ImageEmbedder): model to embed images
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = device
    if embedder_name == "ResNet50Embedder":
        spec = embedder_config[embedder_name]
        return ResNet50Embedder(spec, device)
    elif embedder_name == "CLIPEmbedder":
        spec = embedder_config[embedder_name]
        return CLIPEmbedder(spec, device)
    elif embedder_name == "DETREmbedder":
        spec = embedder_config[embedder_name]
        return DETREmbedder(spec, device)
    elif embedder_name == "BLIPEmbedder":
        spec = embedder_config[embedder_name]
        return BLIPEmbedder(spec, device)
    elif embedder_name == "BLIP2Embedder":
        spec = embedder_config[embedder_name]
        return BLIP2Embedder(spec, device)
    elif embedder_name == "ViT_B_16Embedder":
        spec = embedder_config[embedder_name]
        return ViTEmbedder(spec, device)
    else:
        raise NotImplementedError(f"Embedder {embedder_name} not implemented.")


# NOT USED
# noinspection PyShadowingNames
def create_image_embeddings(
        images: torch.Tensor,
        embedder_list: List[str],
        embedder_config: dict,
) -> Dict[str, torch.Tensor]:
    """
    Create embeddings for a list of images using a list of embedders.

    Args:
        images (torch.Tensor): images to embed
        embedder_list (list): list of embedder names
        embedder_config (dict): dictionary of embedder specifications

    Returns:
        embeddings (dict): dictionary of embeddings
    """
    embeddings = {}
    for embedder_name in embedder_list:
        print(f"Model {embedder_name}")
        image_embedder = embedder_from_spec(embedder_config[embedder_name])
        embedding = image_embedder.embed(images)
        embeddings[embedder_config[embedder_name]["embedding_name"]] = embedding
    return embeddings


def _check_image_tensor_dimensions(images: torch.Tensor):
    """
    Check if image tensor is in (n_samples, n_channels, height, width) format.
    If not, convert it to that format.

    Args:
        images (torch.Tensor): (n_samples, n_channels, height, width)

    Returns:
        images (torch.Tensor): (n_samples, n_channels, height, width)
    """
    try:
        assert images.shape[1] == 3
    except AssertionError:
        print("Images must be in (n_samples, n_channels, height, width) format.")
        if images.shape[3] == 3:
            images = images.permute(0, 3, 1, 2)
        else:
            raise ValueError("Cannot interpret the dimensions of this image tensor."
                             "Check if the shape is (n_samples, n_channels, height, width).")
    return images


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", type=str, default="ResNet50Embedder")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    args.embedder = "BLIP2Embedder"
    args.device = "cuda"
    print('Embedder:', args.embedder)
    embedder = embedder_from_spec(args.embedder, args.device)
    print(embedder)

    # images = torch.rand(32, 3, 224, 224)
    with open("config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load video,
    video_path = './data/stimulus/short_downsampled.avi'
    clip = VideoFileClip(video_path)
    frames = clip.iter_frames()
    images = np.array([frame for frame in frames])
    audio = clip.audio

    # images to tensor
    images = torch.from_numpy(images)

    results = embedder.embed(images)
    # save results as text in csv files via panadas
    import pandas as pd
    dataframe = pd.DataFrame(results)
    dataframe.to_csv("./annotations/blip_results.csv")