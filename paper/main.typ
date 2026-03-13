#set page(
  paper: "us-letter",
  margin: (x: 1in, y: 1in),
  fill: rgb("#FAEBD7"),
)

#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

// Title block
#align(center)[
  #text(size: 18pt, weight: "bold")[Quip: Quantization-Aware Image-Text Pre-training]
  #v(0.5em)
  #text(size: 11pt)[Nathan Leroy]
  #v(0.2em)
  #text(size: 10pt, fill: rgb("#555"))[Qdrant --- March 2026 --- Draft]
  #v(0.3em)
  #line(length: 40%, stroke: 0.5pt + rgb("#999"))
]

#v(1em)

= Abstract

Vector quantization is essential for production-scale retrieval, cutting memory costs by 4--32$times$. While text-only quantization-aware training (QAT) has shown that INT8 embeddings can match float precision, multimodal CLIP-family models degrade significantly under post-hoc quantization. We introduce *Quip* (Quantized Image-text Pre-training), a method that adds lightweight per-modality quantization heads on top of a frozen or fine-tuned CLIP backbone, training the model end-to-end with $floor(127 dot tanh(x) + 0.5)$ quantization and straight-through gradient estimation. Early results on Flickr30k show that Quip retains strong retrieval quality at INT8 precision after training on only 1,000 images, suggesting that native quantization-aware multimodal embeddings are feasible.

= Introduction

CLIP-family models learn a shared embedding space for images and text via contrastive pre-training. At inference time, users frequently quantize these float32 vectors to INT8 or binary representations to reduce memory and accelerate search. However, post-hoc quantization of multimodal embeddings incurs measurable retrieval degradation --- binary-quantized CLIP retains only 87--93% of float performance in optimistic settings.

Quip addresses this by inserting a learned quantization head (Linear $arrow$ LayerNorm $arrow$ $tanh$ $arrow$ $floor$) after each modality's CLIP projection, training the entire model with quantized embeddings in the loss. Gradients flow through the non-differentiable $floor$ via a straight-through estimator (STE). The contrastive loss (InfoNCE with false-negative masking at $tau = 0.02$) operates directly on the INT8 cosine similarities, so the model learns representations that are natively quantization-friendly.

#figure(
  image("figs/quip_overview.svg", width: 95%),
  caption: [
    *Quip architecture.* A pretrained CLIP backbone produces float embeddings, which are passed through per-modality quantization heads ($Q_t$, $Q_v$) to produce INT8 embeddings. The quantization head applies Linear $arrow$ LayerNorm $arrow$ $floor(127 dot tanh(dot) + 0.5)$ with STE gradients. Contrastive loss is computed on the quantized cosine similarities.
  ],
)

= Preliminary Results

We initialize Quip from `openai/clip-vit-base-patch32` and train on a 1,000-image subset of Flickr30k for 5 epochs (batch size 64, cosine LR schedule, $"lr" = 5 times 10^(-4)$). The CLIP backbone is frozen for the first 50 steps to warm up the quantization heads. We evaluate on the full Flickr30k test split (1,000 images, 5,000 captions).

#figure(
  table(
    columns: 7,
    align: (left, center, center, center, center, center, center),
    stroke: none,
    table.hline(stroke: 1pt),
    table.header(
      [*Model*], [*i2t R\@1*], [*R\@5*], [*R\@10*], [*t2i R\@1*], [*R\@5*], [*R\@10*],
    ),
    table.hline(stroke: 0.5pt),
    [CLIP ViT-B/32 (float32)], [77.7], [95.3], [98.4], [59.3], [84.6], [90.8],
    [Quip ViT-B/32 (INT8)], [60.4], [87.8], [94.4], [51.2], [81.8], [89.8],
    table.hline(stroke: 1pt),
  ),
  caption: [
    *Flickr30k retrieval results.* Quip INT8 retains 78% of CLIP's i2t R\@1 and 86% of t2i R\@1 after training on only 1,000 images. The R\@10 gap is notably smaller (94.4 vs 98.4 for i2t, 89.8 vs 90.8 for t2i), indicating the quantized representations preserve coarse-grained ranking well.
  ],
)

#v(1em)

These early numbers are promising: with less than 1% of typical CLIP training data, the quantization heads already learn a mapping that largely preserves retrieval quality through the INT8 bottleneck. We expect significant improvements from scaling to larger training sets (50k--400k images), longer training schedules, and backbone fine-tuning with more data.

= Next Steps

- Scale training to full Flickr30k + COCO (150k+ images)
- Evaluate binary (post-hoc sign) embeddings
- Benchmark against post-hoc quantized CLIP at INT8 and binary
- Ablate InfoNCE vs sigmoid loss under STE gradients
- Test with larger backbones (ViT-L/14, ViT-G/14)