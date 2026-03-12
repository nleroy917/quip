"""
Quip — Quantization-aware Image-text Pre-training model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import CLIPModel, PreTrainedModel
from transformers.utils import ModelOutput

from .config_quip import QuipConfig



# internal straight-through gradient estimator for floor quantization
class _STEFloor(torch.autograd.Function):
    """
    floor() in the forward pass, identity in the backward pass.

    `floor()` is non differentiable, so we use the straight-through estimator (STE) to allow gradients to flow through the quantization operation.
    """

    @staticmethod
    def forward(ctx, x):
        return x.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


ste_floor = _STEFloor.apply


def quantize_int8(x: torch.Tensor, scale: int = 127) -> torch.Tensor:
    r"""floor(scale * tanh(x) + 0.5) with STE on floor. Output \in {-scale, ..., scale}."""
    return ste_floor(scale * torch.tanh(x) + 0.5)


def binarize(x: torch.Tensor) -> torch.Tensor:
    """
    Post-hoc binarization: sign(tanh(x)). Returns {-1, +1}.
    """
    return torch.sign(torch.tanh(x)).clamp(min=-1.0)  # clamp handles exact zeros


# quantization head per modality: projects CLIP features to quantization-suitable space
class QuantizationHead(nn.Module):
    """
    Projects backbone features into a distribution suited for tanh quantization.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.linear(x))


# model output helper class
@dataclass
class QuipOutput(ModelOutput):
    loss: torch.Tensor | None = None
    image_embeds: torch.Tensor | None = None      # quantized (or raw if quant_mode="none")
    text_embeds: torch.Tensor | None = None        # quantized (or raw if quant_mode="none")
    image_embeds_raw: torch.Tensor | None = None   # pre-quantization (after head, before tanh)
    text_embeds_raw: torch.Tensor | None = None


# core model class
class QuipModel(PreTrainedModel):
    config_class = QuipConfig

    def __init__(self, config: QuipConfig):
        super().__init__(config)

        # We don't load the CLIP backbone here — it's loaded in from_pretrained_clip
        # or when HF calls from_pretrained (which restores all weights including clip).
        self.clip = CLIPModel(CLIPModel.config_class())  # placeholder, overwritten by from_pretrained_clip
        self._init_quip_layers(config)

    def _init_quip_layers(self, config: QuipConfig):
        """
        Initialize quantization heads, temperature, and loss-related params.
        """
        self.image_quant_head = QuantizationHead(config.clip_projection_dim, config.quant_embed_dim)
        self.text_quant_head = QuantizationHead(config.clip_projection_dim, config.quant_embed_dim)

        if config.learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(config.temperature)))
        else:
            self.register_buffer("log_temperature", torch.tensor(math.log(config.temperature)))

    @classmethod
    def from_pretrained_clip(
        cls,
        clip_model_name_or_path: str,
        quip_config_overrides: dict | None = None,
        **kwargs,
    ) -> "QuipModel":
        """
        Create a Quip model initialized from a pretrained CLIP checkpoint.

        Args:
            clip_model_name_or_path: HuggingFace model ID or local path for a CLIPModel.
            quip_config_overrides: Dict of QuipConfig fields to override (e.g. quant_mode, loss_type).
        """
        clip = CLIPModel.from_pretrained(clip_model_name_or_path, **kwargs)
        clip_cfg = clip.config

        # Build QuipConfig from the CLIP config dimensions
        quip_kwargs = dict(
            clip_model_name_or_path=clip_model_name_or_path,
            vision_hidden_size=clip_cfg.vision_config.hidden_size,
            text_hidden_size=clip_cfg.text_config.hidden_size,
            clip_projection_dim=clip_cfg.projection_dim,
            quant_embed_dim=clip_cfg.projection_dim,  # default to CLIP's projection dim
        )
        if quip_config_overrides:
            quip_kwargs.update(quip_config_overrides)

        config = QuipConfig(**quip_kwargs)
        model = cls.__new__(cls)
        PreTrainedModel.__init__(model, config)

        model.clip = clip
        model._init_quip_layers(config)
        return model

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def _get_clip_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run CLIP vision encoder -> projection. Returns [B, projection_dim]."""
        out = self.clip.get_image_features(pixel_values=pixel_values)
        # Newer transformers returns BaseModelOutputWithPooling, not a tensor
        if not isinstance(out, torch.Tensor):
            out = out.pooler_output
        return out

    def _get_clip_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run CLIP text encoder -> projection. Returns [B, projection_dim]."""
        out = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        if not isinstance(out, torch.Tensor):
            out = out.pooler_output
        return out

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        quantize: bool = True,
    ) -> torch.Tensor:
        """
        Image features through the quant head. Optionally quantized.
        """
        proj = self._get_clip_image_features(pixel_values)
        raw = self.image_quant_head(proj)
        if not quantize or self.config.quant_mode == "none":
            return raw
        return quantize_int8(raw, scale=self.config.quant_scale)

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        quantize: bool = True,
    ) -> torch.Tensor:
        """
        Text features through the quant head. Optionally quantized.
        """
        proj = self._get_clip_text_features(input_ids, attention_mask)
        raw = self.text_quant_head(proj)
        if not quantize or self.config.quant_mode == "none":
            return raw
        return quantize_int8(raw, scale=self.config.quant_scale)

    def get_image_features_binary(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Post-hoc binary image embeddings: sign(tanh(quant_head(clip_proj))).
        """
        proj = self._get_clip_image_features(pixel_values)
        raw = self.image_quant_head(proj)
        return binarize(raw)

    def get_text_features_binary(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Post-hoc binary text embeddings: sign(tanh(quant_head(clip_proj))).
        """
        proj = self._get_clip_text_features(input_ids, attention_mask)
        raw = self.text_quant_head(proj)
        return binarize(raw)

    def _infonce_loss(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Symmetric InfoNCE with false-negative masking on quantized cosine sims.
        """
        # normalize for cosine similarity
        image_norm = F.normalize(image_embeds.float(), dim=-1)
        text_norm = F.normalize(text_embeds.float(), dim=-1)

        # cosine similarity matrix [B, B], scaled by temperature
        logits = (image_norm @ text_norm.t()) / self.temperature

        # false-negative masking: suppress negatives within margin of the positive
        with torch.no_grad():
            sim = image_norm @ text_norm.t()
            pos_sim = sim.diag().unsqueeze(1)  # [B, 1]
            fn_mask = (sim - pos_sim).abs() < self.config.false_negative_margin
            # Keep the diagonal (true positives)
            fn_mask.fill_diagonal_(False)

        # set masked logits to large negative value
        logits = logits.masked_fill(fn_mask, -1e9)

        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2

    def _sigmoid_loss(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        SigLIP-style pairwise sigmoid loss on quantized cosine sims.
        """
        image_norm = F.normalize(image_embeds.float(), dim=-1)
        text_norm = F.normalize(text_embeds.float(), dim=-1)

        logits = (image_norm @ text_norm.t()) / self.temperature
        B = logits.size(0)

        # Target: +1 on diagonal, -1 off-diagonal
        labels = 2 * torch.eye(B, device=logits.device) - 1
        return F.logsigmoid(labels * logits).sum() / (-B)

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_loss: bool = True,
    ) -> QuipOutput:
        # compute CLIP projected features
        image_proj = self._get_clip_image_features(pixel_values) if pixel_values is not None else None
        text_proj = self._get_clip_text_features(input_ids, attention_mask) if input_ids is not None else None

        # run through quantization heads
        image_raw = self.image_quant_head(image_proj) if image_proj is not None else None
        text_raw = self.text_quant_head(text_proj) if text_proj is not None else None

        # quantize projections
        if self.config.quant_mode == "int8":
            image_embeds = quantize_int8(image_raw, self.config.quant_scale) if image_raw is not None else None
            text_embeds = quantize_int8(text_raw, self.config.quant_scale) if text_raw is not None else None
        elif self.config.quant_mode == "binary":
            image_embeds = binarize(image_raw) if image_raw is not None else None
            text_embeds = binarize(text_raw) if text_raw is not None else None
        else:
            image_embeds = image_raw
            text_embeds = text_raw

        # compute loss if necessary
        loss = None
        if return_loss and image_embeds is not None and text_embeds is not None:
            if self.config.loss_type == "infonce":
                loss = self._infonce_loss(image_embeds, text_embeds)
            elif self.config.loss_type == "sigmoid":
                loss = self._sigmoid_loss(image_embeds, text_embeds)

        return QuipOutput(
            loss=loss,
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            image_embeds_raw=image_raw,
            text_embeds_raw=text_raw,
        )