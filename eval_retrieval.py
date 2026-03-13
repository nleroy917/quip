"""
Image-text retrieval evaluation for Quip and baseline CLIP models.

Usage:
    # Vanilla CLIP baseline on Flickr30k
    python eval_retrieval.py --clip_baseline --datasets flickr30k

    # Trained Quip at all quant levels vs CLIP baseline
    python eval_retrieval.py \
        --quip_checkpoint ./quip-flickr-run/checkpoint-100 \
        --clip_baseline \
        --quant_modes float int8 binary \
        --datasets flickr30k coco
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoProcessor

from quip import QuipModel
from eval import (
    CLIPEmbedder,
    QuipEmbedder,
    DATASET_LOADERS,
    evaluate_retrieval,
    print_results_table,
)


def main():
    parser = argparse.ArgumentParser(description="Quip retrieval evaluation")
    parser.add_argument("--quip_checkpoint", type=str, default=None, help="Path to trained Quip checkpoint dir")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name (used for processor and baseline)")
    parser.add_argument("--clip_baseline", action="store_true", help="Also evaluate vanilla CLIP as baseline")
    parser.add_argument("--datasets", nargs="+", default=["flickr30k"], choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--quant_modes", nargs="+", default=["int8"], choices=["float", "int8", "binary"])
    parser.add_argument("--max_images", type=int, default=None, help="Subset each dataset to N images (for quick gut checks)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    # Load datasets
    eval_datasets = {}
    for ds_name in args.datasets:
        print(f"Loading {ds_name}...")
        eval_datasets[ds_name] = DATASET_LOADERS[ds_name](max_images=args.max_images)
        ds = eval_datasets[ds_name]
        print(f"  {len(ds.images)} images, {len(ds.texts)} texts")

    # Build list of embedders to evaluate
    embedders = []

    if args.clip_baseline:
        embedders.append(CLIPEmbedder(args.clip_model, device=args.device))

    if args.quip_checkpoint:
        processor = AutoProcessor.from_pretrained(args.clip_model)
        model = QuipModel.from_pretrained_clip(args.clip_model)

        # Load trained weights
        import safetensors.torch
        ckpt_path = os.path.join(args.quip_checkpoint, "model.safetensors")
        if os.path.exists(ckpt_path):
            state_dict = safetensors.torch.load_file(ckpt_path)
        else:
            ckpt_path = os.path.join(args.quip_checkpoint, "pytorch_model.bin")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {ckpt_path}")

        for qm in args.quant_modes:
            embedders.append(QuipEmbedder(model, processor, quant_mode=qm, device=args.device))

    if not embedders:
        parser.error("Provide at least one of --quip_checkpoint or --clip_baseline")

    # Run evals
    all_results = {}
    for embedder in embedders:
        print(f"\nEvaluating: {embedder.name}")
        all_results[embedder.name] = {}
        for ds_name, dataset in eval_datasets.items():
            results = evaluate_retrieval(embedder, dataset, batch_size=args.batch_size)
            all_results[embedder.name][ds_name] = results

    print_results_table(all_results)


if __name__ == "__main__":
    main()


# python eval_retrieval.py \
#     --quip_checkpoint ./quip-flickr-run/checkpoint-785 \
#     --clip_baseline \
#     --quant_modes int8 \
#     --max_images 1000 \
#     --datasets flickr30k