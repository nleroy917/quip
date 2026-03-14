"""
Run Quip retrieval evaluation on Modal (GPU, full datasets).

Usage:
    # Download eval datasets to the volume (one-time)
    modal run eval_modal.py::download_data

    # Evaluate a specific checkpoint against all baselines on Flickr30k
    modal run eval_modal.py --run-name <comet-run-name>

    # Evaluate a specific checkpoint dir
    modal run eval_modal.py --checkpoint-dir quip-coco-run/<run-name>/checkpoint-333

    # Also evaluate on COCO
    modal run eval_modal.py --run-name <comet-run-name> --datasets flickr30k coco
"""

import modal

app = modal.App("quip-eval")

# same volumes as training
data_volume = modal.Volume.from_name("quip-data", create_if_missing=True)
output_volume = modal.Volume.from_name("quip-output", create_if_missing=True)

DATA_DIR = "/data"
OUTPUT_DIR = "/output"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch",
        "transformers",
        "datasets<4.0.0",
        "accelerate",
        "safetensors",
        "pillow",
    )
    .add_local_dir("quip", "/root/quip")
    .add_local_dir("eval", "/root/eval")
)


@app.function(
    image=image,
    volumes={DATA_DIR: data_volume},
    timeout=3600,
)
def download_data():
    import os
    os.environ["HF_HOME"] = DATA_DIR

    from datasets import load_dataset

    print("Downloading nlphuji/flickr30k test split...")
    ds = load_dataset("nlphuji/flickr30k", split="test")
    print(f"Downloaded flickr30k test: {len(ds)} rows")

    print("Downloading jxie/coco_captions test split...")
    ds = load_dataset("jxie/coco_captions", split="test")
    print(f"Downloaded coco test: {len(ds)} rows")

    data_volume.commit()
    print("Done — eval datasets cached to volume")


@app.function(
    image=image,
    gpu="A10",
    volumes={DATA_DIR: data_volume, OUTPUT_DIR: output_volume},
    timeout=3600,
)
def evaluate(
    run_name: str | None = None,
    checkpoint_dir: str | None = None,
    datasets: list[str] = ["flickr30k"],
    quant_modes: list[str] = ["int8"],
    batch_size: int = 256,
    clip_model: str = "openai/clip-vit-base-patch32",
):
    import os
    import sys

    os.environ["HF_HOME"] = DATA_DIR
    sys.path.insert(0, "/root")

    import torch
    from transformers import AutoProcessor

    from quip import QuipModel
    from eval import (
        CLIPEmbedder,
        CLIPQuantizedEmbedder,
        CLIPBinaryEmbedder,
        QuipEmbedder,
        DATASET_LOADERS,
        evaluate_retrieval,
        print_results_table,
    )

    device = "cuda" # running on GPU (see above)

    # resolve checkpoint path
    if run_name:
        ckpt_base = os.path.join(OUTPUT_DIR, "quip-coco-run", run_name)
    elif checkpoint_dir:
        ckpt_base = os.path.join(OUTPUT_DIR, checkpoint_dir)
    else:
        raise ValueError("Provide either --run-name or --checkpoint-dir")

    # find the latest checkpoint subdir (highest step number)
    candidates = sorted(
        [d for d in os.listdir(ckpt_base) if d.startswith("checkpoint-")],
        key=lambda d: int(d.split("-")[-1]),
    )
    if candidates:
        ckpt_path = os.path.join(ckpt_base, candidates[-1])
        print(f"Using latest checkpoint: {ckpt_path}")
    else:
        # naybe the path itself is a checkpoint dir
        ckpt_path = ckpt_base
        print(f"Using checkpoint dir: {ckpt_path}")

    # load datasets
    eval_datasets = {}
    for ds_name in datasets:
        print(f"Loading {ds_name}...")
        eval_datasets[ds_name] = DATASET_LOADERS[ds_name]()
        ds = eval_datasets[ds_name]
        print(f"  {len(ds.images)} images, {len(ds.texts)} texts")

    embedders = []
    embedders.append(CLIPEmbedder(clip_model, device=device)) # vanilla CLIP baseline
    embedders.append(CLIPQuantizedEmbedder(clip_model, device=device)) # post-hoc quantized CLIP baseline
    embedders.append(CLIPBinaryEmbedder(clip_model, device=device)) # post-hoc binary quantized CLIP baseline
    processor = AutoProcessor.from_pretrained(clip_model)
    model = QuipModel.from_pretrained_clip(clip_model) # initialize with CLIP weights, then load trained checkpoint weights on top

    import safetensors.torch
    weights_path = os.path.join(ckpt_path, "model.safetensors")
    if os.path.exists(weights_path):
        state_dict = safetensors.torch.load_file(weights_path)
    else:
        weights_path = os.path.join(ckpt_path, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {weights_path}")

    for qm in quant_modes:
        embedders.append(QuipEmbedder(model, processor, quant_mode=qm, device=device))

    # run evals
    all_results = {}
    for embedder in embedders:
        print(f"\nEvaluating: {embedder.name}")
        all_results[embedder.name] = {}
        for ds_name, dataset in eval_datasets.items():
            results = evaluate_retrieval(embedder, dataset, batch_size=batch_size)
            all_results[embedder.name][ds_name] = results

    print_results_table(all_results)


@app.local_entrypoint()
def main(
    run_name=None,
    checkpoint_dir=None,
    datasets=["coco", "flickr30k"],
    quant_modes=["int8"],
    batch_size=256,
    clip_model="openai/clip-vit-base-patch32",
):
    evaluate.remote(
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        datasets=datasets,
        quant_modes=quant_modes,
        batch_size=batch_size,
        clip_model=clip_model,
    )
