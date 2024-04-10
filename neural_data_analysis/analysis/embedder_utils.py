import torch
from neural_data_analysis.analysis.ImageEmbedder import (
    VGG16Embedder,
    ResNet50Embedder,
    CLIPEmbedder,
    DETREmbedder,
    DINOEmbedder,
    BLIPEmbedder,
    BLIP2Embedder,
    ViTEmbedder,
)
from neural_data_analysis.analysis.TextEmbedder import SGPTEmbedder
import yaml
import numpy as np
from moviepy.editor import VideoFileClip

embedder_config = {
    "VGG16Embedder": {
        "embedding_name": "vgg16",
        "batch_size": 64,
    },
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
        "batch_size": 1,
        "model": "Salesforce/blip2-opt-2.7b",
        "processor": "Salesforce/blip2-opt-2.7b",
    },
    "DINOEmbedder": {
        "embedding_name": "dino",
        "embedding_description": "DINO: Emerging Properties in Self-Supervised Vision Transformers",
        "batch_size": 1,
        "model": "facebook/dinov2-base",
        "processor": "facebook/dinov2-base",
    },
    "ViT_B_16Embedder": {
        "embedding_name": "vit",
        "embedding_description": "Vision Transformer Base 16",
        "batch_size": 64,
    },
    "SGPTEmbedder": {
        "embedding_name": "sgpt",
    },
    "rgbhsvl": {
        "embedding_name": "rgbhsvl",
    },
    "gist": {
        "embedding_name": "gist",
    },
    "moten": {
        "embedding_name": "moten",
    },
    "face_regressors": {
        "embedding_name": "face_regressors",
    },
}


def embedder_from_spec(
    embedder_name: str, device: str = None
) -> (
    ResNet50Embedder
    | CLIPEmbedder
    | DETREmbedder
    | BLIPEmbedder
    | BLIP2Embedder
    | DINOEmbedder
    | ViTEmbedder
    | SGPTEmbedder
):
    """
    Create an embedder from a specification dictionary.

    Args:
        embedder_name (str): name of Embedder to create
        device (str): device to use for embedding (cpu or cuda)

    Returns:
        embedder (ImageEmbedder): model to embed images
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = device
    if embedder_name == "VGG16Embedder":
        spec = embedder_config[embedder_name]
        return VGG16Embedder(spec, device)
    elif embedder_name == "ResNet50Embedder":
        spec = embedder_config[embedder_name]
        return ResNet50Embedder(spec, device)
    elif embedder_name == "CLIPEmbedder":
        spec = embedder_config[embedder_name]
        return CLIPEmbedder(spec, device)
    elif embedder_name == "DETREmbedder":
        spec = embedder_config[embedder_name]
        return DETREmbedder(spec, device)
    elif embedder_name == "DINOEmbedder":
        spec = embedder_config[embedder_name]
        return DINOEmbedder(spec, device)
    elif embedder_name == "BLIPEmbedder":
        spec = embedder_config[embedder_name]
        return BLIPEmbedder(spec, device)
    elif embedder_name == "BLIP2Embedder":
        spec = embedder_config[embedder_name]
        return BLIP2Embedder(spec, device)
    elif embedder_name == "ViT_B_16Embedder":
        spec = embedder_config[embedder_name]
        return ViTEmbedder(spec, device)
    elif embedder_name == "SGPTEmbedder":
        spec = embedder_config[embedder_name]
        return SGPTEmbedder(spec, device)
    else:
        raise NotImplementedError(f"Embedder {embedder_name} not implemented.")


# NOT USED
# noinspection PyShadowingNames
def create_image_embeddings(
    images: torch.Tensor,
    embedder_list: list[str],
    embedder_config: dict,
) -> dict[str, torch.Tensor]:
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", type=str, default="ResNet50Embedder")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    # args.embedder = "BLIP2Embedder"
    # args.embedder = "VGG16Embedder"
    args.embedder = "SGPTEmbedder"
    args.device = "cuda"

    print("Embedder:", args.embedder)
    embedder = embedder_from_spec(args.embedder, args.device)
    print(embedder)
    # images = torch.rand(32, 3, 224, 224)
    with open("config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load video,
    video_path = "./data/stimulus/short_downsampled.avi"
    clip = VideoFileClip(video_path)
    frames = clip.iter_frames()
    images = np.array([frame for frame in frames])
    audio = clip.audio

    # images to tensor
    images = torch.from_numpy(images)

    results = embedder.embed(images)
    # save results as text in csv files via pandas
    import pandas as pd

    dataframe = pd.DataFrame(results)
    dataframe.to_csv("./annotations/blip_results.csv")
