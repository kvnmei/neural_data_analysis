from abc import ABC, abstractmethod
import openl3
import soundfile as sf
from wandb.wandb_torch import torch


class AudioEmbedder(ABC):
    """
        Embeds input audio by an encoding model.

        Attributes:
            config (dict): configuration dictionary
            device (torch.device): device to use for embedding

        Methods:
            preprocess(self, encoder)
            embed(self, encoder)
    """

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.batch_size = config.get("batch_size", 64)
        self.encoder = None

    @abstractmethod
    def preprocess(self, audio) -> dict:
        pass

    @abstractmethod
    def embed(self, audio):
        pass

class OpenL3AudioEmbedder(AudioEmbedder):
    def __init__(self, config: dict, device: torch.device):
        super().__init__(config, device)

    def embed(self, audio_path:str):
        audio, sr = sf.read(audio_path)
        embedding, timestamps = openl3.get_embedding(audio, sr, content_type = "env")
        return embedding





