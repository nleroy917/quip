"""Quip — Quantization-aware Image-text Pre-training configuration."""

from transformers import PretrainedConfig


class QuipConfig(PretrainedConfig):
    model_type = "quip"

    def __init__(
        self,
        # Backbone
        clip_model_name_or_path: str = "openai/clip-vit-base-patch16",
        vision_hidden_size: int = 768,
        text_hidden_size: int = 512,
        clip_projection_dim: int = 512,
        # Quantization head
        quant_embed_dim: int = 512,
        quant_mode: str = "int8",  # "int8" | "binary" | "none"
        quant_scale: int = 127,  # floor(scale * tanh(x) + 0.5)
        # Loss
        loss_type: str = "infonce",  # "infonce" | "sigmoid"
        temperature: float = 0.02,
        learnable_temperature: bool = False,
        false_negative_margin: float = 0.1,
        # Matryoshka (optional)
        matryoshka_dims: list[int] | None = None,
        # Distillation (optional, off by default per research plan)
        distillation_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clip_model_name_or_path = clip_model_name_or_path
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.clip_projection_dim = clip_projection_dim
        self.quant_embed_dim = quant_embed_dim
        self.quant_mode = quant_mode
        self.quant_scale = quant_scale
        self.loss_type = loss_type
        self.temperature = temperature
        self.learnable_temperature = learnable_temperature
        self.false_negative_margin = false_negative_margin
        self.matryoshka_dims = matryoshka_dims
        self.distillation_weight = distillation_weight