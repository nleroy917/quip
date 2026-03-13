"""
Retrieval metrics and evaluation runner.
"""

from __future__ import annotations

import torch

from .datasets import RetrievalDataset
from .embedders import MultimodalEmbedder


def compute_recall_at_k(
    scores: torch.Tensor,
    ground_truth: dict[int, list[int] | int],
    k_values: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """
    Compute Recall@K from a similarity matrix.

    Args:
        scores: [num_queries, num_candidates] similarity matrix.
        ground_truth: query_idx -> candidate_idx (int) or list of valid candidate indices.
        k_values: list of K values to compute recall at.
    """
    results = {}
    num_queries = scores.size(0)

    for k in k_values:
        _, topk_indices = scores.topk(k, dim=1)
        hits = 0
        for q_idx in range(num_queries):
            gt = ground_truth[q_idx]
            if isinstance(gt, int):
                gt = [gt]
            if any(g in topk_indices[q_idx].tolist() for g in gt):
                hits += 1
        results[f"R@{k}"] = hits / num_queries * 100

    return results


def evaluate_retrieval(
    embedder: MultimodalEmbedder,
    dataset: RetrievalDataset,
    batch_size: int = 64,
) -> dict:
    """
    Run full retrieval eval: image->text and text->image Recall@{1,5,10}.
    """
    print(f"  Encoding {len(dataset.images)} images...")
    image_embeds = embedder.encode_images(dataset.images, batch_size=batch_size)

    print(f"  Encoding {len(dataset.texts)} texts...")
    text_embeds = embedder.encode_texts(dataset.texts, batch_size=batch_size)

    # [num_images, num_texts]
    scores = image_embeds @ text_embeds.t()

    print("  Computing image->text recall...")
    i2t = compute_recall_at_k(scores, dataset.image_to_text)

    print("  Computing text->image recall...")
    t2i = compute_recall_at_k(scores.t(), dataset.text_to_image)

    return {"i2t": i2t, "t2i": t2i}


def print_results_table(all_results: dict[str, dict[str, dict]]):
    """
    Print a formatted comparison table.
    """
    datasets = list(next(iter(all_results.values())).keys())
    models = list(all_results.keys())

    for ds_name in datasets:
        print(f"\n{'=' * 70}")
        print(f"  {ds_name}")
        print(f"{'=' * 70}")

        header = f"{'Model':<25} | {'i2t R@1':>7} {'R@5':>6} {'R@10':>6} | {'t2i R@1':>7} {'R@5':>6} {'R@10':>6}"
        print(header)
        print("-" * len(header))

        for model_name in models:
            r = all_results[model_name][ds_name]
            i2t, t2i = r["i2t"], r["t2i"]
            print(
                f"{model_name:<25} | {i2t['R@1']:>7.1f} {i2t['R@5']:>6.1f} {i2t['R@10']:>6.1f} "
                f"| {t2i['R@1']:>7.1f} {t2i['R@5']:>6.1f} {t2i['R@10']:>6.1f}"
            )
