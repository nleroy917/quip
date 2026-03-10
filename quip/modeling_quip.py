"""
Quip: Quantization-aware Image-text Pre-training

Architecture sketch for a SigLIP-based multimodal embedding model
that natively outputs INT8 / binary quantized embeddings.

Key design decisions:
  1. Compose, don't inherit — QuipModel wraps SigLIP's vision and text
     encoders but replaces the projection + normalization layer with a
     quantization-aware head.
  2. Separate quant heads per modality — vision and text encoder outputs
     have different activation distributions; per-modality heads let each
     learn its own quantization mapping.
  3. STE (Straight-Through Estimator) for floor — following pplx-embed,
     we use floor(127·tanh(x) + 0.5) with STE to backprop through the
     non-differentiable floor operation.
  4. Cosine similarity on quantized embeddings — matches pplx-embed's
     approach and works natively with Qdrant's distance metrics.
  5. Configurable loss: InfoNCE (pplx-embed style) or sigmoid pairwise
     (SigLIP style). InfoNCE is default to match proven QAT recipe,
     sigmoid is available as an experimental alternative for multimodal.
  6. Binary quantization is post-hoc — pplx-embed finds training-free
     binarization works with minimal loss. No need for STE on sign().

HuggingFace integration:
  - QuipConfig is a PretrainedConfig subclass → push_to_hub / from_pretrained
  - QuipModel is a PreTrainedModel subclass → AutoModel registration
  - from_siglip_pretrained() class method loads a pretrained SigLIP checkpoint

Corrections after reviewing pplx-embed paper (arXiv:2602.11151v2):
  - Quantization uses floor(127·tanh(x) + 0.5), NOT round(127·tanh(x))
  - Output range is {-127,...,127}, not {-128,...,127}
  - Cosine similarity on quantized embeddings, not scaled dot product
  - pplx-embed uses InfoNCE with false-negative masking, not sigmoid loss
  - pplx-embed uses NO distillation loss — pure contrastive on quantized
  - Binary quantization is post-hoc sign(), no QAT needed for binary
  - Temperature τ=0.02 for pair training, τ=0.03 for triplet training
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    PreTrainedModel,
    SiglipConfig,
    SiglipVisionModel,
    SiglipTextModel,
    AutoModel,
    AutoConfig,
)
from transformers.utils import ModelOutput
from .config_quip import QuipConfig


# ste primitives
class StraightThroughFloor(torch.autograd.Function):
    """
    Floor with straight-through gradient estimation.

    This is pplx-embed's approach. they compute:
        floor(127 · tanh(mean_pool) + 0.5)

    The +0.5 before floor makes this behave like round-to-nearest for
    most values, but floor has a cleaner STE story: it's a piecewise-
    constant function with well-defined behavior at every point, whereas
    round() has ambiguous behavior at exact half-integers.

    Forward: y = floor(x)
    Backward: dy/dx = 1  (straight-through)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.floor()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output  # straight-through


ste_floor = StraightThroughFloor.apply

# quantization head with learned projection + LayerNorm, followed by tanh and STE floor
class QuantizationHead(nn.Module):
    """
    Projects encoder features to a quantized embedding space.

    Architecture:
        Linear(input_dim → output_dim) → LayerNorm → tanh → quantize

    The quantization step follows pplx-embed's formula:
        floor(127 · tanh(x) + 0.5)

    producing integer values in {-127, ..., 127} — signed 8-bit.

    Why tanh instead of L2 norm?
    L2 normalization maps to the unit hypersphere, but rounding a unit
    vector to INT8 wastes most of the [-127, 127] range (values cluster
    near 0 because ||v||=1 means each component is ~1/sqrt(d)).
    Tanh naturally bounds outputs to (-1, 1) and distributes mass more
    uniformly across the range after scaling by 127.

    Why a learned Linear+LayerNorm here when pplx-embed has none?
    pplx-embed applies tanh directly to mean-pooled token embeddings.
    Their backbone IS the feature extractor. But for Quip, we're
    starting from SigLIP's pooler_output which was trained with L2-norm
    in mind, not tanh. The Linear+LN adapts the distribution so tanh
    doesn't immediately saturate. Think of it as replacing SigLIP's
    original projection head with a quant-aware one.

    Binary quantization:
    Following pplx-embed's finding that "training-free post-hoc
    binarization can be applied with minimal performance loss," binary
    mode simply applies sign() to the pre-quantization tanh output.
    No STE needed for binary — just train with INT8 QAT, then binarize
    at inference time.
    """

    def __init__(self, input_dim: int, output_dim: int, quant_mode: str = "int8"):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.ln = nn.LayerNorm(output_dim)
        self.quant_mode = quant_mode

    def forward(
        self,
        x: torch.Tensor,
        quantize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            quantized: The quantized embedding (INT8 or float)
            float_emb: The pre-quantization tanh embedding (for optional distillation)
        """
        x = self.proj(x)
        x = self.ln(x)
        float_emb = torch.tanh(x)  # \in (-1, 1)

        if not quantize or self.quant_mode == "float":
            return float_emb, float_emb

        if self.quant_mode == "int8":
            # pplx-embed formula: floor(127 · tanh(x) + 0.5)
            # the +0.5 makes floor behave like round-to-nearest.
            # Output range: {-127, ..., 127}
            quantized = ste_floor(float_emb * 127.0 + 0.5)
            quantized = quantized.clamp(-127, 127)
        else:
            raise ValueError(f"Unknown quant_mode: {self.quant_mode}")

        return quantized, float_emb

    def binarize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Post-hoc binary quantization (no STE, inference only).

        Follows pplx-embed: bin(x) = 1 if x >= 0, else -1.
        Applied to the pre-quantization tanh output.

        Usage:
            _, float_emb = quant_head(encoder_output, quantize=False)
            binary_emb = quant_head.binarize(float_emb)
        """
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


# helper output class
@dataclass
class QuipOutput(ModelOutput):
    """Typed output container — plays nicely with HF's pipeline infra."""
    loss: Optional[torch.Tensor] = None
    contrastive_loss: Optional[torch.Tensor] = None
    distillation_loss: Optional[torch.Tensor] = None
    image_embeds: Optional[torch.Tensor] = None       # quantized
    text_embeds: Optional[torch.Tensor] = None         # quantized
    image_embeds_float: Optional[torch.Tensor] = None  # pre-quantization tanh
    text_embeds_float: Optional[torch.Tensor] = None   # pre-quantization tanh


# main model class
class QuipModel(PreTrainedModel):
    """
    Quantization-aware multimodal embedding model.

    Architecture overview:

        ┌──────────────┐     ┌──────────────┐
        │  SigLIP       │     │  SigLIP       │
        │  VisionModel  │     │  TextModel    │
        └──────┬───────┘     └──────┬───────┘
               │ pooler_output       │ pooler_output
               ▼                     ▼
        ┌──────────────┐     ┌──────────────┐
        │  Vision       │     │  Text         │
        │  Quant Head   │     │  Quant Head   │
        │  Lin→LN→tanh  │     │  Lin→LN→tanh  │
        │  →floor(STE)  │     │  →floor(STE)  │
        └──────┬───────┘     └──────┬───────┘
               │ INT8 embeds         │ INT8 embeds
               ▼                     ▼
        ┌────────────────────────────────────┐
        │  cosine similarity                 │
        │  → InfoNCE loss (default)          │
        │    with false-negative masking     │
        │  OR sigmoid pairwise (optional)    │
        └────────────────────────────────────┘

    Key differences from pplx-embed:
    - pplx-embed is text-only; Quip is multimodal (dual-tower vision+text)
    - pplx-embed applies tanh directly to mean-pooled tokens; Quip adds a
      learned projection since we're adapting SigLIP's pooler output
    - Quip uses separate quant heads per modality (vision and text have
      different activation distributions)
    - Quip optionally supports SigLIP's sigmoid pairwise loss as an
      alternative to InfoNCE

    Key similarities with pplx-embed:
    - Same quantization formula: floor(127·tanh(x) + 0.5) with STE
    - Same use of cosine similarity on quantized embeddings
    - Same approach to binary: post-hoc sign(), no QAT needed
    - INT8 quantization active during all training stages
    """

    config_class = QuipConfig
    # tell hf which param names are the backbone (for differential LR)
    _no_split_modules = ["SiglipVisionModel", "SiglipTextModel"]

    def __init__(self, config: QuipConfig):
        super().__init__(config)

        siglip_config = SiglipConfig(**config.siglip_config)

        # core backbone models
        self.vision_model = SiglipVisionModel(siglip_config.vision_config)
        self.text_model = SiglipTextModel(siglip_config.text_config)

        # quantization-aware projection heads (per-modality) ---
        vision_hidden = siglip_config.vision_config.hidden_size
        text_hidden = siglip_config.text_config.hidden_size

        self.vision_quant_head = QuantizationHead(
            input_dim=vision_hidden,
            output_dim=config.quant_embed_dim,
            quant_mode=config.quant_mode,
        )
        self.text_quant_head = QuantizationHead(
            input_dim=text_hidden,
            output_dim=config.quant_embed_dim,
            quant_mode=config.quant_mode,
        )

        # loss-specific parameters
        if config.loss_type == "sigmoid":
            # SigLIP-style: learnable temperature + bias
            self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init))
            self.logit_bias = nn.Parameter(torch.tensor(config.logit_bias_init))

        if config.freeze_backbone:
            self._freeze_backbone()

        self.post_init()

    def _freeze_backbone(self):
        """
        Freeze SigLIP encoders — only train quant heads + loss params.
        """
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False

    @classmethod
    def from_siglip_pretrained(
        cls,
        siglip_model_name_or_path: str,
        quip_config_overrides: Optional[dict] = None,
        **kwargs,
    ) -> "QuipModel":
        """Initialize Quip from a pretrained SigLIP checkpoint.

        Usage:
            model = QuipModel.from_siglip_pretrained(
                "google/siglip2-so400m-patch14-384",
                quip_config_overrides={
                    "quant_mode": "int8",
                    "quant_embed_dim": 768,
                    "loss_type": "infonce",
                    "temperature": 0.02,
                },
            )

        Loads SigLIP's vision and text encoder weights. The quant heads
        are randomly initialized (they must be trained).
        """
        from transformers import SiglipModel as _SiglipModel

        siglip = _SiglipModel.from_pretrained(siglip_model_name_or_path, **kwargs)
        siglip_config = siglip.config

        overrides = quip_config_overrides or {}
        config = QuipConfig(siglip_config=siglip_config, **overrides)
        model = cls(config)

        # copy pretrained encoder weights
        model.vision_model.load_state_dict(siglip.vision_model.state_dict())
        model.text_model.load_state_dict(siglip.text_model.state_dict())

        return model

    
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        quantize: bool = True,
    ) -> torch.Tensor:
        """
        Extract (optionally quantized) image embeddings.
        """
        vision_out = self.vision_model(pixel_values=pixel_values)
        pooled = vision_out.pooler_output
        quantized, _ = self.vision_quant_head(pooled, quantize=quantize)
        return quantized

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        quantize: bool = True,
    ) -> torch.Tensor:
        """
        Extract (optionally quantized) text embeddings.
        """
        text_out = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = text_out.pooler_output
        quantized, _ = self.text_quant_head(pooled, quantize=quantize)
        return quantized

    def get_image_features_binary(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract binary image embeddings (post-hoc, inference only).
        """
        vision_out = self.vision_model(pixel_values=pixel_values)
        _, float_emb = self.vision_quant_head(vision_out.pooler_output, quantize=False)
        return self.vision_quant_head.binarize(float_emb)

    def get_text_features_binary(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract binary text embeddings (post-hoc, inference only).
        """
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        _, float_emb = self.text_quant_head(text_out.pooler_output, quantize=False)
        return self.text_quant_head.binarize(float_emb)

    # training forward pass ---
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> QuipOutput:
        # encoder forward pass
        vision_out = self.vision_model(pixel_values=pixel_values)
        text_out = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # pool via mean pooling + LayerNorm (SigLIP's default) — this is what the quant heads are designed for
        vision_pooled = vision_out.pooler_output
        text_pooled = text_out.pooler_output

        # quantize
        image_quant, image_float = self.vision_quant_head(vision_pooled)
        text_quant, text_float = self.text_quant_head(text_pooled)

        loss = None
        contrastive_loss = None
        distillation_loss = None

        if return_loss:
            # primary contrastive loss (on quantized embeddings) ---
            if self.config.loss_type == "infonce":
                contrastive_loss = self._infonce_loss(image_quant, text_quant)
            elif self.config.loss_type == "sigmoid":
                contrastive_loss = self._sigmoid_contrastive_loss(image_quant, text_quant)
            else:
                raise ValueError(f"Unknown loss_type: {self.config.loss_type}")

            loss = contrastive_loss

            # optional: distillation loss on pre-quantization float embeddings ---
            # pplx-embed does NOT use this — they train purely on quantized
            # embeddings. We include it as an optional regularizer for cases
            # where the multimodal alignment is fragile during early QAT.
            # Default weight is 0.0 (off).
            if self.config.distill_weight > 0:
                distillation_loss = self._distillation_loss(image_float, text_float)
                loss = loss + self.config.distill_weight * distillation_loss

        return QuipOutput(
            loss=loss,
            contrastive_loss=contrastive_loss,
            distillation_loss=distillation_loss,
            image_embeds=image_quant,
            text_embeds=text_quant,
            image_embeds_float=image_float,
            text_embeds_float=text_float,
        )

    # loss functions ---
    def _infonce_loss(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE loss with false-negative masking, following pplx-embed.

        pplx-embed (Section 2.3) uses:
        - Cosine similarity on quantized embeddings
        - Temperature τ (default 0.02)
        - False-negative masking: if a negative's similarity to the query
          exceeds the positive pair's similarity by more than 0.1, mask it
          (it's likely a false negative, not a true negative)
        - Both in-batch documents AND in-batch queries as negatives

        For Quip (multimodal), we adapt this slightly:
        - "queries" = image embeddings, "documents" = text embeddings
        - We include image-image negatives (analogous to query-query negs)
        """
        # cosine similarity on quantized embeddings
        image_norm = F.normalize(image_embeds.float(), dim=-1)
        text_norm = F.normalize(text_embeds.float(), dim=-1)

        # image-to-text similarities (B x B)
        sim_i2t = image_norm @ text_norm.t() / self.config.temperature
        # image-to-image similarities for cross-modal negatives
        sim_i2i = image_norm @ image_norm.t() / self.config.temperature

        batch_size = image_embeds.shape[0]

        # positive similarities (diagonal)
        pos_sim = sim_i2t.diag()  # (B,)

        # --- false-negative masking (pplx-embed Section 2.3) ---
        # mask negatives whose similarity is within `margin` of the positive.
        # this prevents punishing the model for near-duplicates in the batch.
        margin = self.config.false_neg_margin / self.config.temperature
        # m(d_j) = 1 if sim(q_i, d_j) <= sim(q_i, d_i) + margin, else 0
        text_mask = (sim_i2t <= pos_sim.unsqueeze(1) + margin).float()
        img_mask = (sim_i2i <= pos_sim.unsqueeze(1) + margin).float()

        # don't mask the positive itself; don't use self-similarity for img-img
        text_mask.fill_diagonal_(0)  # positive handled separately in numerator
        img_mask.fill_diagonal_(0)   # exclude self-similarity

        # run infoNCE with masked negatives:
        # log[ exp(pos) / (exp(pos) + Σ_neg masked_exp(neg)) ]
        pos_exp = pos_sim.exp()  # (B,)

        # masked negatives from text
        neg_text = (sim_i2t.exp() * text_mask).sum(dim=1)  # (B,)
        # masked negatives from other images (query-query negatives)
        neg_img = (sim_i2i.exp() * img_mask).sum(dim=1)    # (B,)

        loss = -torch.log(pos_exp / (pos_exp + neg_text + neg_img + 1e-8)).mean()
        return loss

    def _sigmoid_contrastive_loss(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        SigLIP-style pairwise sigmoid loss (alternative to InfoNCE).

        Unlike InfoNCE, this computes binary cross-entropy for every
        (image, text) pair independently — no softmax over the batch.

        Potential advantages for multimodal QAT:
        - Per-pair gradients are independent → less noisy STE updates
        - No batch-size sensitivity (InfoNCE implicitly depends on B)
        - Matches SigLIP's original training objective

        Potential disadvantages:
        - Less proven for quantization-aware training (pplx-embed uses InfoNCE)
        - May need larger batches to see enough negatives
        """
        # cosine similarity on quantized embeddings
        image_norm = F.normalize(image_embeds.float(), dim=-1)
        text_norm = F.normalize(text_embeds.float(), dim=-1)

        logits = image_norm @ text_norm.t()
        logits = logits * self.logit_scale.exp() + self.logit_bias

        batch_size = image_embeds.shape[0]
        labels = 2 * torch.eye(batch_size, device=logits.device) - 1

        loss = -F.logsigmoid(labels * logits).mean()
        return loss

    def _distillation_loss(
        self,
        image_float: torch.Tensor,
        text_float: torch.Tensor,
    ) -> torch.Tensor:
        """Optional distillation loss on pre-quantization float embeddings.

        NOTE: pplx-embed does NOT use this. They train purely on quantized
        embeddings and it works fine for text. For multimodal, cross-modal
        alignment may be more fragile, so we keep this as an option.

        When enabled, this regularizes the float representations to maintain
        good cross-modal alignment independently of quantization quality.
        """
        image_norm = F.normalize(image_float, dim=-1)
        text_norm = F.normalize(text_float, dim=-1)

        sim = image_norm @ text_norm.t() / self.config.temperature

        batch_size = image_float.shape[0]
        labels = torch.arange(batch_size, device=sim.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss


# auto-reg
AutoConfig.register("quip", QuipConfig)
AutoModel.register(QuipConfig, QuipModel)


# example usages
if __name__ == "__main__":
    # --- Option 1: From pretrained SigLIP ---
    # model = QuipModel.from_siglip_pretrained(
    #     "google/siglip2-so400m-patch14-384",
    #     quip_config_overrides={
    #         "quant_mode": "int8",
    #         "quant_embed_dim": 768,
    #         "loss_type": "infonce",
    #         "temperature": 0.02,
    #     },
    # )

    # from config (for testing)
    config = QuipConfig(
        quant_mode="int8",
        quant_embed_dim=768,
        loss_type="infonce",
        temperature=0.02,
    )
    model = QuipModel(config)

    # dummy forward pass
    batch_size = 4
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 32000, (batch_size, 16))

    output = model(pixel_values=pixel_values, input_ids=input_ids)

    print(f"Loss:             {output.loss:.4f}")
    print(f"Image embeds:     {output.image_embeds.shape} dtype={output.image_embeds.dtype}")
    print(f"Text embeds:      {output.text_embeds.shape} dtype={output.text_embeds.dtype}")
    print(f"Image range:      [{output.image_embeds.min():.0f}, {output.image_embeds.max():.0f}]")
    print(f"Float range:      [{output.image_embeds_float.min():.4f}, {output.image_embeds_float.max():.4f}]")

    # int8 inference
    with torch.no_grad():
        img_int8 = model.get_image_features(pixel_values)
        txt_int8 = model.get_text_features(input_ids)
        # cosine similarity on INT8
        sim = F.cosine_similarity(img_int8.float(), txt_int8.float(), dim=-1)
        print(f"\nINT8 cosine sim:  {sim}")

    # binary inference (post-hoc, no QAT needed)
    with torch.no_grad():
        img_bin = model.get_image_features_binary(pixel_values)
        txt_bin = model.get_text_features_binary(input_ids)
        # binary similarity ∝ (d - 2·hamming_distance)
        sim_bin = F.cosine_similarity(img_bin, txt_bin, dim=-1)
        print(f"Binary cosine sim: {sim_bin}")