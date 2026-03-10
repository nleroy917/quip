from __future__ import annotations

from typing import Optional

from transformers import (
    PretrainedConfig,
    SiglipConfig,
)


class QuipConfig(PretrainedConfig):
    """
    Why a separate config instead of subclassing SiglipConfig?
    Because Quip adds new hyperparameters (quant_mode, quant_embed_dim, etc.)
    that don't belong in SiglipConfig's namespace, and we want SiglipConfig
    to stay a nested, self-contained blob so we can swap in SigLIP2, etc.
    later without touching QuipConfig's schema.
    """
    model_type = "quip"

    def __init__(
        self,
        siglip_config: Optional[dict] = None,
        quant_mode: str = "int8",           # "int8" | "float"
        quant_embed_dim: int = 768,         # output embedding dimension
        loss_type: str = "infonce",         # "infonce" | "sigmoid"
        temperature: float = 0.02,          # τ for contrastive loss (pplx-embed uses 0.02)
        false_neg_margin: float = 0.1,      # margin for false-negative masking (InfoNCE)
        distill_weight: float = 0.0,        # 0.0 = off (pplx-embed uses none). Optional regularizer.
        logit_scale_init: float = 2.6592,   # ln(1/0.07), only used if loss_type="sigmoid"
        logit_bias_init: float = -10.0,     # SigLIP-style bias, only used if loss_type="sigmoid"
        freeze_backbone: bool = False,      # freeze SigLIP encoders
        **kwargs,
    ):
        super().__init__(**kwargs)
        # store SigLIP config as a nested dict — PretrainedConfig serializes
        # this cleanly to/from JSON for push_to_hub.
        if siglip_config is None:
            siglip_config = SiglipConfig().to_dict()
        elif isinstance(siglip_config, SiglipConfig):
            siglip_config = siglip_config.to_dict()
        self.siglip_config = siglip_config
        self.quant_mode = quant_mode
        self.quant_embed_dim = quant_embed_dim
        self.loss_type = loss_type
        self.temperature = temperature
        self.false_neg_margin = false_neg_margin
        self.distill_weight = distill_weight
        self.logit_scale_init = logit_scale_init
        self.logit_bias_init = logit_bias_init
        self.freeze_backbone = freeze_backbone