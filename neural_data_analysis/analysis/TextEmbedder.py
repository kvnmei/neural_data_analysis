from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
from typing import Dict, List, Protocol
from abc import ABC, abstractmethod


class TextEmbedder(ABC):
    """
    Embeds input images by an encoding model.

    Attributes:
        config (dict): configuration dictionary
        device (torch.device): device to use for embedding

    Methods:
        preprocess(self, encoder)
        embed(self, encoder)

    """

    def __init__(self, config: dict, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.batch_size = config.get("batch_size", 64)
        self.encoder = None

    @abstractmethod
    def embed(self, sentences: list[str]) -> torch.Tensor:
        pass


class SGPTEmbedder(TextEmbedder):
    def __init__(self, config: dict, device: torch.device) -> None:
        super().__init__(config, device)

    def embed(self, sentences: list[str]) -> torch.Tensor:
        model = SentenceTransformer("Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit")
        embeddings = model.encode(
            sentences, batch_size=self.batch_size, show_progress_bar=True
        )
        return embeddings
