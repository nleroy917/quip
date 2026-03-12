"""
Gut check: load a pretrained CLIP inside Quip, grab embeddings from both,
and compare cosine similarities to verify the backbone loaded correctly.
"""

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel
from transformers.image_utils import load_image

from quip import QuipModel, show_image

CLIP_MODEL = "openai/clip-vit-base-patch32"

# Load processor + vanilla CLIP for reference
processor = AutoProcessor.from_pretrained(CLIP_MODEL)
clip = CLIPModel.from_pretrained(CLIP_MODEL)
clip.eval()

# Load Quip wrapping the same CLIP
quip = QuipModel.from_pretrained_clip(CLIP_MODEL)
quip.eval()

# Inputs
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)
texts = ["a photo of a two cats laying down", "a photo of a dog", "a photo of a bird"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# vanilla clip
with torch.no_grad():
    clip_image_embeds = clip.get_image_features(pixel_values=inputs["pixel_values"]).pooler_output
    clip_text_embeds = clip.get_text_features(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
    ).pooler_output
    clip_image_norm = F.normalize(clip_image_embeds, dim=-1)
    clip_text_norm = F.normalize(clip_text_embeds, dim=-1)
    clip_sims = clip_image_norm @ clip_text_norm.t()

print("=== Vanilla CLIP cosine similarities ===")
for i, t in enumerate(texts):
    print(f"  {t}: {clip_sims[0, i]:.4f}")

# reach inside quip to get the CLIP embeddings and verify they match the above (sanity check for correct loading)
with torch.no_grad():
    quip_clip_image_embeds = quip.clip.get_image_features(pixel_values=inputs["pixel_values"]).pooler_output
    quip_clip_text_embeds = quip.clip.get_text_features(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
    ).pooler_output
    quip_clip_image_norm = F.normalize(quip_clip_image_embeds, dim=-1)
    quip_clip_text_norm = F.normalize(quip_clip_text_embeds, dim=-1)
    quip_clip_sims = quip_clip_image_norm @ quip_clip_text_norm.t()

print("\n=== Quip inner CLIP cosine similarities (should match above) ===")
for i, t in enumerate(texts):
    print(f"  {t}: {quip_clip_sims[0, i]:.4f}")

# --- Verify they match ---
diff = (clip_sims - quip_clip_sims).abs().max().item()
print(f"\n  Max difference: {diff:.2e}")
assert diff < 1e-5, f"Mismatch! Max diff = {diff}"
print("  PASS: backbone loaded correctly")

# quip quantized embeddings (random quant heads, just show they work, since we have a learned projection layer
# that is randomly initialized, we expect these to be noisy but not NaN or Inf or anything crazy, which would indicate a loading issue)
with torch.no_grad():
    quip_image_q = quip.get_image_features(inputs["pixel_values"], quantize=True)
    quip_text_q = quip.get_text_features(
        inputs["input_ids"], inputs["attention_mask"], quantize=True,
    )
    quip_image_qn = F.normalize(quip_image_q.float(), dim=-1)
    quip_text_qn = F.normalize(quip_text_q.float(), dim=-1)
    quip_q_sims = quip_image_qn @ quip_text_qn.t()

print("\n=== Quip INT8 cosine similarities (untrained quant heads — expect noise) ===")
for i, t in enumerate(texts):
    print(f"  {t}: {quip_q_sims[0, i]:.4f}")
