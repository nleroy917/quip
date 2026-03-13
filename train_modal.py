"""
Train Quip on COCO captions via Modal.

Usage:
    # Download the dataset to the volume (one-time)
    modal run train_modal.py::download_data

    # Run training
    modal run train_modal.py::train
"""

import modal

app = modal.App("quip-train")

# persistent volumes: one for HF dataset cache, one for checkpoints
data_volume = modal.Volume.from_name("quip-data", create_if_missing=True)
output_volume = modal.Volume.from_name("quip-output", create_if_missing=True)

DATA_DIR = "/data"
OUTPUT_DIR = "/output"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "comet_ml[gpu]",
        "python-dotenv",
        "safetensors",
        "pillow",
    )
    .add_local_dir("quip", "/root/quip")
)


# downloading + caching the dataset on the volume is separate from training so that we can do it once and then iterate on training without re-downloading, which is a bit faster and more convenient for development
@app.function(
    image=image,
    volumes={DATA_DIR: data_volume},
    timeout=3600,
)
def download_data():
    import os
    os.environ["HF_HOME"] = DATA_DIR

    from datasets import load_dataset

    print("Downloading jxie/coco_captions train split...")
    ds_train = load_dataset("jxie/coco_captions", split="train")
    print(f"Downloaded train: {len(ds_train)} rows")

    print("Downloading jxie/coco_captions validation split...")
    ds_val = load_dataset("jxie/coco_captions", split="validation")
    print(f"Downloaded validation: {len(ds_val)} rows")

    # Force cache to flush to volume
    data_volume.commit()
    print("Done — dataset cached to volume")


# actual training function, which mounts the volume with the cached dataset and the output volume for checkpoints, and trains the model
@app.function(
    image=image,
    gpu="A100",
    volumes={DATA_DIR: data_volume, OUTPUT_DIR: output_volume},
    timeout=7200, # 2 hours should be enough for a few epochs, and we can always increase if needed
    secrets=[modal.Secret.from_name("comet-api-key")],
)
def train():
    import os
    import sys

    # Point HF cache at the persistent volume
    os.environ["HF_HOME"] = DATA_DIR

    # Make quip importable
    sys.path.insert(0, "/root")

    import comet_ml
    comet_ml.login(project_name="quip")

    from datasets import load_dataset
    from transformers import AutoProcessor

    from quip import QuipModel, QuipTrainer, QuipTrainingArguments
    from quip.data import QuipImageTextDataset

    CLIP_MODEL = "openai/clip-vit-base-patch32"
    SEED = 42

    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(CLIP_MODEL)
    model = QuipModel.from_pretrained_clip(
        CLIP_MODEL,
        quip_config_overrides={
            "quant_mode": "int8",
            "loss_type": "infonce",
            "temperature": 0.02,
        },
    )

    print("Loading COCO captions from volume cache...")
    ds_train = load_dataset("jxie/coco_captions", split="train")
    ds_val = load_dataset("jxie/coco_captions", split="validation")
    print(f"Loaded {len(ds_train)} train rows, {len(ds_val)} val rows")

    train_dataset = QuipImageTextDataset.from_coco_captions(ds_train, processor)
    eval_dataset = QuipImageTextDataset.from_coco_captions(ds_val, processor)
    print(f"Training on {len(train_dataset)} images, validating on {len(eval_dataset)} images")

    training_args = QuipTrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "quip-coco-run"),
        num_train_epochs=3,
        per_device_train_batch_size=128,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        # fp16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="epoch",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to=["comet_ml"],
        seed=SEED,
        freeze_backbone_steps=200,
    )

    model = model.train()
    trainer = QuipTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # Flush checkpoints to volume
    output_volume.commit()
    print(f"Checkpoints saved to {OUTPUT_DIR}/quip-coco-run")