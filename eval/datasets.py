"""
Retrieval evaluation datasets in a common format.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class RetrievalDataset:
    """
    Holds loaded eval data in a common format.
    """

    name: str
    images: list                         # list of PIL images
    texts: list[str]                     # flat list of all captions
    image_to_text: dict[int, list[int]]  # image_idx -> [text_idx, ...]
    text_to_image: dict[int, int]        # text_idx -> image_idx


def load_flickr30k(max_images: int | None = None) -> RetrievalDataset:
    """
    Load the Flickr30k test split.
    """
    ds = load_dataset("nlphuji/flickr30k", split="test")
    if max_images is not None:
        ds = ds.select(range(min(max_images, len(ds))))

    images, texts = [], []
    image_to_text, text_to_image = {}, {}

    for img_idx, row in enumerate(ds):
        images.append(row["image"])
        text_indices = []
        for cap in row["caption"]:
            t_idx = len(texts)
            texts.append(cap)
            text_indices.append(t_idx)
            text_to_image[t_idx] = img_idx
        image_to_text[img_idx] = text_indices

    return RetrievalDataset("Flickr30k", images, texts, image_to_text, text_to_image)


def load_coco_karpathy(max_images: int | None = None) -> RetrievalDataset:
    """
    Load COCO Karpathy test split (jxie/coco_captions — images inline, one row per caption).
    """
    ds = load_dataset("jxie/coco_captions", split="test")

    # group by cocoid since the dataset is flattened
    grouped = OrderedDict()
    for row in ds:
        cid = row["cocoid"]
        if cid not in grouped:
            grouped[cid] = {"image": row["image"], "captions": []}
        grouped[cid]["captions"].append(row["caption"])

    images, texts = [], []
    image_to_text, text_to_image = {}, {}

    for img_idx, (_, group) in enumerate(grouped.items()):
        if max_images is not None and img_idx >= max_images:
            break
        images.append(group["image"])
        text_indices = []
        for cap in group["captions"]:
            t_idx = len(texts)
            texts.append(cap)
            text_indices.append(t_idx)
            text_to_image[t_idx] = img_idx
        image_to_text[img_idx] = text_indices

    return RetrievalDataset("COCO-Karpathy", images, texts, image_to_text, text_to_image)


DATASET_LOADERS = {
    "flickr30k": load_flickr30k,
    "coco": load_coco_karpathy,
}
