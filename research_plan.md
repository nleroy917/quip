# Research Proposal: Quantization-Aware Multimodal Embedding Models

**Author:** Nathan Leroy, Qdrant
**Date:** March 2026
**Status:** Draft — Internal

---

## Motivation

Vector quantization is table stakes for production-scale retrieval. At Qdrant we routinely see customers compress billion-vector indices to INT8 or binary representations, cutting memory costs by 4–32×. For text-only embeddings this works remarkably well — Perplexity's recent **pplx-embed** models demonstrate that quantization-aware training (QAT) can produce INT8 embeddings that match full-precision retrieval quality, and binary embeddings that trail by only ~1.5 nDCG points.

Multimodal embeddings are a different story. CLIP-family models (OpenCLIP, SigLIP, SigLIP 2) experience **significant retrieval degradation** when their float32 vectors are post-hoc quantized. Marqo's experiments show binary-quantized CLIP retains only 87–93% of float performance even after adding a pseudo-quantization training loss — and that is the *optimistic* case with careful activation tuning. Naïve post-hoc binarization can drop to ~80% recall retention.

This creates a critical gap for large e-commerce marketplaces that need both multimodal (image + text) search *and* aggressive vector compression. Today they must choose between retrieval quality and infrastructure cost. We propose to close that gap by training multimodal contrastive embedding models that **natively output quantized representations**, following the paradigm Perplexity pioneered for text — but extending it to the vision-language domain.

---

## Research Questions

1. **Baseline degradation:** How much do leading multimodal models (OpenCLIP, SigLIP 2, Qwen3-VL-Embedding) actually degrade at INT8 and binary precision across standard retrieval benchmarks?
2. **Architecture design:** What architectural and loss-function modifications enable a dual-tower vision-language model to produce quantization-friendly embeddings without sacrificing float-precision quality?
3. **Training recipe:** Can we adapt the pplx-embed recipe — `floor(127·tanh(mean) + 0.5)` with STE, InfoNCE loss at τ=0.02, false-negative masking — to contrastive multimodal pre-training on a dual-tower vision-language backbone?
4. **Competitive quality:** Can our quantized multimodal embeddings match or exceed post-hoc quantized baselines while delivering measurable throughput and storage advantages?

---

## Phase 1 — Baseline Performance Characterization

**Goal:** Establish the quantitative cost of post-hoc quantization on current multimodal models.

**Models to evaluate:**

| Model | Params | Source |
|---|---|---|
| OpenCLIP ViT-L/14 | 428M | OpenAI / LAION |
| SigLIP 2 So400m/14 | 400M | Google |
| SigLIP 2 ViT-B/16 | 86M | Google |
| Qwen3-VL-Embedding-2B | 2B | Alibaba |

**Quantization schemes:** FP32 (baseline), INT8 scalar, Binary (sign bit).

**Datasets & metrics:**

- **COCO 5K** — image↔text retrieval, Recall@1/5/10
- **Flickr30K** — image↔text retrieval, Recall@1/5/10
- **Marqo GS-E-Commerce** — product search with in-domain / out-of-domain splits, MRR@10 and nDCG@10
- **MMEB-V2** (subset) — broad multimodal embedding benchmark

**Deliverables:** A table showing absolute and relative recall drops per model × quantization scheme × dataset. This becomes the "wall" our trained models need to beat.

---

## Phase 2 — Literature Review & Architecture Design

### Prior art to synthesize

**Text-only quantization-aware embeddings (pplx-embed, arXiv:2602.11151v2):**
Perplexity's pplx-embed models use diffusion-based continued pre-training on Qwen3 (converting the causal LM to a bidirectional encoder), followed by multi-stage contrastive fine-tuning (pair → contextual → triplet, with SLERP model merging). During all contrastive stages, embeddings are quantized to INT8 via the formula `floor(127·tanh(mean_pool(v)) + 0.5)`, with gradients flowing through a straight-through estimator (STE) on the floor operation. Critically, they use **InfoNCE loss** (not sigmoid) with a false-negative masking mechanism (mask negatives within 0.1 cosine similarity of the positive), temperature τ=0.02, and **cosine similarity** on the quantized INT8 vectors. This yields INT8 vectors that match FP32 on MTEB retrieval (69.66 vs. 69.60 nDCG@10 at 4B scale). For binary, they find **post-hoc binarization** (sign function on the pre-quantization mean-pooled embeddings) works with minimal loss — no QAT needed for binary. The 4B model loses only ~1.5 nDCG points from INT8 to binary, though the 0.6B model loses 2–4.4 points, suggesting higher embedding dimensions (2560 vs. 1024) provide more resilience to binarization.

**Multimodal binarization:**
Marqo's "Learn to Binarize CLIP" adds a pseudo-quantization loss alongside the standard contrastive objective during fine-tuning. They find sigmoid activation outperforms tanh for binarization, retaining 87–93% of float performance depending on the evaluation split. Crucially, float performance itself is preserved at ~99.7% when the quantization loss is co-trained — so the model doesn't sacrifice generality.

**SigLIP 2 architecture:**
Google's SigLIP 2 replaces softmax-based InfoNCE with a sigmoid pairwise loss, enabling per-pair optimization without batch-global normalization. Note that pplx-embed chose InfoNCE over sigmoid for their QAT recipe and achieved excellent results — so the sigmoid loss advantage for QAT is unproven. However, SigLIP's per-pair gradients may still interact better with the noisy STE gradients from floor quantization (each pair's gradient update is independent, reducing variance). This is an empirical question we should test.

**Qwen3-VL-Embedding:**
Alibaba's newest multimodal embedding model (Jan 2026) uses multi-stage contrastive pre-training with Matryoshka Representation Learning, achieving SOTA on MMEB-V2 at 77.8. This represents the current best-in-class for general multimodal retrieval — but does **not** natively produce quantized embeddings.

### Proposed architecture — "Quip" (Quantization-aware Image-text Pre-training)

We propose a **dual-tower SigLIP-style architecture** (codename: **Quip**) with the following modifications for quantization awareness:

1. **Backbone:** Start from a SigLIP 2 So400m/14 checkpoint (or Qwen3-VL-2B for a larger variant). Use the vision and text encoders directly, bypassing SigLIP's original L2-normalized projection heads.
2. **Per-modality quantization heads:** Replace SigLIP's projection + L2 norm with learned `Linear → LayerNorm → tanh → floor(·*127 + 0.5)` heads, one per modality. The Linear+LN adapts the distribution from SigLIP's pooler (trained for L2 norm) to one suited for tanh. pplx-embed has no learned projection (they apply tanh directly to mean-pooled tokens), but we need one because we're adapting a pretrained SigLIP backbone, not training from scratch. Separate heads per modality are important because vision and text encoder outputs have very different activation distributions — a shared head forces a single quantization mapping to handle both.
3. **Quantization formula:** `floor(127·tanh(x) + 0.5)` with STE on the floor operation, following pplx-embed exactly. Output range is `{-127, ..., 127}`, representable as signed INT8. Quantization is active during all training stages.
4. **Contrastive loss on quantized embeddings:** Default is **InfoNCE** (matching pplx-embed's proven recipe) with false-negative masking (margin=0.1), temperature τ=0.02, and **cosine similarity** on the quantized vectors. We will also experiment with SigLIP's sigmoid pairwise loss as an alternative — its per-pair gradient independence may interact better with the noisy STE updates in a multimodal setting.
5. **No distillation loss by default:** pplx-embed uses zero distillation — pure contrastive training on quantized embeddings. We follow this as the default, but keep distillation as an optional regularizer (weight=0) in case multimodal alignment proves more fragile than text-only during early QAT.
6. **Binary is post-hoc, not trained:** Following pplx-embed's finding that "training-free post-hoc binarization can be applied with minimal performance loss," we apply `sign()` to the pre-quantization tanh embeddings at inference time. No separate binary training phase needed — this simplifies the pipeline and saves ~1–2 weeks of training.
7. **Matryoshka support:** Optionally train with Matryoshka Representation Learning so users can choose embedding dimension at inference time. pplx-embed's contextual model uses this with dimensions [128, 256, 512, 1024, 2048, 2560].
8. **HuggingFace-native:** The model is implemented as a `PreTrainedModel` subclass composing `SiglipVisionModel` + `SiglipTextModel`, with `AutoModel` registration for `from_pretrained()` / `push_to_hub()` support out of the box. See `quip_architecture.py` for the full implementation sketch.

### Key technical risks

- **Embedding dimension vs. binary resilience:** pplx-embed shows their 4B model (dim=2560) loses only ~1.5 nDCG points on binarization, while their 0.6B (dim=1024) loses 2–4.4 points. For Quip, our target dim of 768 (SigLIP So400m default) may be too low for resilient binarization. We may need to project up to 1024+ before quantization, which trades memory savings for quality.
- **Image-side sensitivity:** Vision transformers may be more sensitive to quantization than text encoders due to the continuous nature of pixel features. The per-modality quant heads help, but we may still see asymmetric degradation (image retrieval dropping more than text retrieval). Monitoring per-direction recall (image→text vs. text→image) will be critical.
- **STE gradient bias:** The straight-through estimator introduces gradient bias that accumulates over training. pplx-embed mitigates this with careful learning rate selection (2e-4 for 0.6B, 5e-5 for 4B) and gradient clipping. We should follow their hyperparameter strategy.
- **SigLIP pooler distribution mismatch:** SigLIP's pooler_output was trained with L2 normalization as the downstream operation. Switching to tanh changes the optimal feature distribution. The Linear+LN in our quant head handles this, but early training may be unstable as the head adapts. A warm-up phase with frozen backbone and only quant head training could help.
- **Loss function interaction with STE:** pplx-embed uses InfoNCE and it works. But InfoNCE's softmax denominator means a single bad gradient estimate (from STE) in one pair can affect all other pairs' gradients in the batch. Sigmoid loss avoids this coupling — an important ablation to run.

---

## Phase 3 — Training Data

### Requirements

We need paired (image, text) data at scale, ideally with hard negatives and relevance labels. Three tiers of data:

**Tier 1 — Public pre-training data (warm-up):**
- **LAION-400M / DataComp-1B** — large-scale image-text pairs for initial contrastive warm-up
- **CC12M** — Conceptual Captions for additional diversity

**Tier 2 — Curated retrieval data (contrastive fine-tuning):**
- **MS-COCO** — 118K images, 5 captions each, with hard negatives from cross-modal mining
- **Visual Genome** — dense region descriptions for fine-grained alignment
- **STS-B / SNLI-VE** — cross-modal entailment pairs

**Tier 3 — E-commerce domain data (specialization):**
This is where proprietary data creates differentiation. Options include:
- **Amazon ESCI** (public) — 710K products with images and search queries, including relevance labels. This is the strongest public option.
- **Marqo GS-E-Commerce** — fashion / retail evaluation data with in-domain and out-of-domain splits
- **Partner data** — If Qdrant can arrange data partnerships with e-commerce customers (anonymized product catalogs), this becomes the strongest moat. Even a few hundred thousand labeled product-query pairs in a specific vertical (fashion, electronics, home goods) would significantly boost domain performance.

### Data pipeline

1. Filter and deduplicate public data (DataComp filtering pipeline)
2. Mine hard negatives per batch using the frozen teacher model's similarity scores
3. Augment with cropping, color jitter, and back-translation for text

---

## Phase 4 — Evaluation & Benchmarking

### Head-to-head comparisons

For each model below, we evaluate at FP32, INT8, and Binary precision:

| Model | Type | Notes |
|---|---|---|
| OpenCLIP ViT-L/14 | Baseline (post-hoc quant) | Standard CLIP |
| SigLIP 2 So400m/14 | Baseline (post-hoc quant) | Sigmoid loss |
| Qwen3-VL-Embedding-2B | Baseline (post-hoc quant) | Current SOTA |
| **Quip (ours)** | Quantization-aware | Our model |

### Metrics

**Retrieval quality:**
- Recall@1, Recall@5, Recall@10 (image→text and text→image)
- nDCG@10 and MRR@10 on ranked retrieval tasks
- MMEB-V2 aggregate score

**Efficiency:**
- Memory per vector (bytes) at each precision level
- Indexing throughput (vectors/sec) in Qdrant at each precision
- Query latency (p50, p99) for 1M and 100M index sizes
- Recall@10 vs. memory Pareto curves — the key chart for the narrative

**Quantization retention ratio:**
- `metric_at_INTx / metric_at_FP32` — this is the single number that tells the story. Our target: ≥99% at INT8, ≥95% at binary.

### Evaluation scripts

All evaluation code will be packaged as a reproducible benchmark suite:
- Built on MTEB / MMEB evaluation harness where possible
- Custom Qdrant integration for throughput and latency measurement
- Single `eval.py` entrypoint with config YAML for model/dataset/precision combinations

---

## Timeline (Estimated)

| Phase | Duration | Output |
|---|---|---|
| Phase 1: Baselines | 2–3 weeks | Degradation report |
| Phase 2: Lit review + arch design | 2 weeks (parallel with Phase 1) | Architecture RFC + `quip_architecture.py` |
| Phase 3: Data pipeline | 2–3 weeks | Cleaned training data |
| Training: INT8 Quip model | 4–6 weeks | Quip-INT8 checkpoint (binary is post-hoc, free) |
| Loss ablation: InfoNCE vs. sigmoid | 1 week (parallel) | Ablation report |
| Phase 4: Evaluation | 2 weeks | Benchmark report (INT8 + binary + float) |
| Write-up + open-source prep | 2 weeks | Blog post / paper draft |
| **Total** | **~12–16 weeks** | |

---

## Compute Requirements (Rough Estimate)

- **Contrastive fine-tuning with QAT:** 8× A100-80GB, ~2–4 weeks (pplx-embed trains pair stage for 50K steps at batch size 16,384)
- **Loss ablation runs:** 4× A100-80GB, ~1 week (InfoNCE vs. sigmoid comparison)
- **Evaluation runs:** 1× A100 for embedding generation, Qdrant cluster for retrieval benchmarks
- **Total GPU budget:** ~1,500–3,000 A100-hours (reduced: no separate binary training phase)

---

## Success Criteria

1. **INT8 multimodal embeddings** that retain ≥99% of FP32 Recall@10 on COCO and Flickr30K — matching what pplx-embed achieves for text-only.
2. **Binary multimodal embeddings** that retain ≥95% of FP32 Recall@10, significantly outperforming post-hoc binarized CLIP/SigLIP.
3. **Published Pareto improvement** on the recall-vs-memory curve compared to all post-hoc quantized baselines.
4. **Open-source release** of model weights, training code, and evaluation suite — positioning Qdrant as the authoritative voice on efficient multimodal retrieval.

---

## References

1. Perplexity AI. "pplx-embed: State-of-the-Art Embedding Models for Web-Scale Retrieval." arXiv:2602.11151v2, Feb 2026.
2. Marqo. "Learn to Binarize CLIP for Multimodal Retrieval and Ranking." 2024.
3. Zhai et al. "SigLIP 2: Multilingual Vision-Language Encoders." arXiv:2502.14786, Feb 2025.
4. Qwen Team. "Qwen3-VL-Embedding and Qwen3-VL-Reranker." arXiv:2601.04720, Jan 2026.
5. Bengio et al. "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation." 2013.
6. Amazon. "Benchmarking Image Embeddings for E-Commerce." arXiv:2504.07567, 2025.
7. Marqo. "Matryoshka Representation Learning with CLIP for Multimodal Retrieval and Ranking." 2024.
8. Kusupati et al. "Matryoshka Representation Learning." NeurIPS 2022.
9. Gong et al. "Scaling up Masked Diffusion Models on Text." arXiv, 2025.
10. Nie et al. "Large Language Diffusion Models." arXiv, 2025.