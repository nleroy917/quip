"""
Multimodal embedder interface and implementations for CLIP and Quip.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel

from quip import QuipModel


class MultimodalEmbedder(ABC):
    """
    Abstract interface for any model that embeds images and text.
    """

    name: str

    @abstractmethod
    def encode_images(self, images: list, batch_size: int = 64) -> torch.Tensor:
        """
        Encode a list of PIL images into normalized embeddings [N, D].
        """
        raise NotImplementedError

    @abstractmethod
    def encode_texts(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        """
        Encode a list of strings into normalized embeddings [N, D].
        """
        raise NotImplementedError


class CLIPEmbedder(MultimodalEmbedder):
    """
    Vanilla CLIPModel — baseline for comparison.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device
        self.name = f"CLIP({model_name.split('/')[-1]})"

    @torch.no_grad()
    def encode_images(self, images: list, batch_size: int = 64) -> torch.Tensor:
        all_embeds = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            out = self.model.get_image_features(**inputs)
            embeds = out if isinstance(out, torch.Tensor) else out.pooler_output
            all_embeds.append(F.normalize(embeds.float(), dim=-1).cpu())
        return torch.cat(all_embeds, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.processor(
                text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77,
            ).to(self.device)
            out = self.model.get_text_features(**inputs)
            embeds = out if isinstance(out, torch.Tensor) else out.pooler_output
            all_embeds.append(F.normalize(embeds.float(), dim=-1).cpu())
        return torch.cat(all_embeds, dim=0)


class QuipEmbedder(MultimodalEmbedder):
    """
    Trained QuipModel — evaluable at float, int8, or binary precision.
    """

    def __init__(self, model: QuipModel, processor, quant_mode: str = "int8", device: str = "cpu"):
        self.model = model.to(device).eval()
        self.processor = processor
        self.quant_mode = quant_mode
        self.device = device
        self.name = f"Quip({quant_mode})"

    @torch.no_grad()
    def encode_images(self, images: list, batch_size: int = 64) -> torch.Tensor:
        all_embeds = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            if self.quant_mode == "binary":
                embeds = self.model.get_image_features_binary(inputs["pixel_values"])
            elif self.quant_mode == "int8":
                embeds = self.model.get_image_features(inputs["pixel_values"], quantize=True)
            else:
                embeds = self.model.get_image_features(inputs["pixel_values"], quantize=False)
            all_embeds.append(F.normalize(embeds.float(), dim=-1).cpu())
        return torch.cat(all_embeds, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.processor(
                text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77,
            ).to(self.device)
            if self.quant_mode == "binary":
                embeds = self.model.get_text_features_binary(inputs["input_ids"], inputs.get("attention_mask"))
            elif self.quant_mode == "int8":
                embeds = self.model.get_text_features(inputs["input_ids"], inputs.get("attention_mask"), quantize=True)
            else:
                embeds = self.model.get_text_features(inputs["input_ids"], inputs.get("attention_mask"), quantize=False)
            all_embeds.append(F.normalize(embeds.float(), dim=-1).cpu())
        return torch.cat(all_embeds, dim=0)
