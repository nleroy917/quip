"""
Quick training run: fine-tune Quip on a 1k-image subset of Flickr30k.

Usage:
    python train_flickr.py

Requires: pip install datasets
"""
import random

import torch
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoProcessor
from transformers.image_utils import load_image

from quip import QuipModel, QuipTrainer, QuipTrainingArguments

def sanity_check_eval(model: QuipModel, processor: AutoProcessor):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = load_image(url)
    texts = ["a photo of two cats laying down", "a photo of a dog", "a photo of a bird"]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        img_q = model.get_image_features(inputs["pixel_values"], quantize=True)
        txt_q = model.get_text_features(inputs["input_ids"], inputs["attention_mask"], quantize=True)
        img_qn = F.normalize(img_q.float(), dim=-1)
        txt_qn = F.normalize(txt_q.float(), dim=-1)
        sims = img_qn @ txt_qn.t()
    print("\n=== Post-training Quip INT8 cosine similarities ===")
    for i, t in enumerate(texts):
        print(f"  {t}: {sims[0, i]:.4f}")

CLIP_MODEL = "openai/clip-vit-base-patch32"
NUM_IMAGES = 1000
SEED = 42
OUTPUT_DIR = "./quip-flickr-run"

processor = AutoProcessor.from_pretrained(CLIP_MODEL)
model = QuipModel.from_pretrained_clip(
    CLIP_MODEL,
    quip_config_overrides={
        "quant_mode": "int8",
        "loss_type": "infonce",
        "temperature": 0.02,
    },
)

# pre-training sanity check to verify the CLIP backbone is working and the quant heads produce finite outputs
# the sims will be random since the quant heads are randomly initialized, but they should be finite and not NaN or Inf or anything crazy, which would indicate a loading issue
# this should ideally show really poor sims since the quant heads are random, but just verify they run and produce finite outputs without error before we start training
sanity_check_eval(model, processor)

ds = load_dataset("nlphuji/flickr30k", split="test")  # flickr30k only has a test split on HF
ds = ds.shuffle(seed=SEED).select(range(min(NUM_IMAGES, len(ds))))

print(f"Dataset: {len(ds)} images")
print(f"Sample captions: {ds[0]['caption'][:2]}")

class QuipFlickrDataset(torch.utils.data.Dataset):
    """
    Wraps a HF dataset of (image, caption list) pairs for contrastive training.

    Each __getitem__ picks one random caption per image and returns
    processor-ready tensors.
    """
    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row["image"]
        # pick one random caption from the 5 available
        caption = random.choice(row["caption"])
        encoded = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )
        # Squeeze batch dim (processor returns [1, ...])
        return {k: v.squeeze(0) for k, v in encoded.items()}


train_dataset = QuipFlickrDataset(ds, processor)
training_args = QuipTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=64,
    learning_rate=5e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False,  # we pass custom dict keys
    dataloader_num_workers=4,
    report_to="none",
    seed=SEED,
    # Quip-specific: warm up quant heads for 50 steps with frozen backbone
    freeze_backbone_steps=50,
)

model = model.train()  # set to train mode before creating trainer to ensure correct weight tying behavior
trainer = QuipTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("Starting training...")
trainer.train()
print("Training complete!")

model.eval()
sanity_check_eval(model, processor)

