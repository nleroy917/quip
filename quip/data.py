import random

from torch.utils.data import Dataset


class QuipImageTextDataset(Dataset):
    """
    Generic dataset for contrastive training. Expects a list of
    (image, captions) tuples where captions is a list of strings.

    Each __getitem__ picks one random caption per image.
    """

    def __init__(self, images, captions, processor):
        assert len(images) == len(captions)
        self.images = images
        self.captions = captions
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        caption = random.choice(self.captions[idx]) if isinstance(self.captions[idx], list) else self.captions[idx]

        encoded = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

    @classmethod
    def from_flickr30k(cls, hf_dataset, processor) -> "QuipImageTextDataset":
        """
        Load from nlphuji/flickr30k format (one row per image, caption is a list).
        """
        images = [row["image"] for row in hf_dataset]
        captions = [row["caption"] for row in hf_dataset]
        return cls(images, captions, processor)

    @classmethod
    def from_coco_captions(cls, hf_dataset, processor) -> "QuipImageTextDataset":
        """
        Load from jxie/coco_captions format (one row per caption, grouped by cocoid).

        Builds a lightweight index of (row_indices, captions) per image, then
        reads images lazily via __getitem__ to avoid loading 113k JPEGs upfront.
        """
        return _CocoCaptionsDataset(hf_dataset, processor)


class _CocoCaptionsDataset(Dataset):
    """
    Lazy-loading COCO dataset. Builds a fast index from the caption column
    (strings only, no image decoding), then loads one image at a time in __getitem__.
    """

    def __init__(self, hf_dataset, processor):
        self.hf_dataset = hf_dataset
        self.processor = processor

        # Build index using only the lightweight columns — no image decoding
        cocoids = hf_dataset["cocoid"]
        captions = hf_dataset["caption"]

        image_index = {}  # cocoid -> {"row_idx": int, "captions": [str]}
        for row_idx, (cid, cap) in enumerate(zip(cocoids, captions)):
            if cid not in image_index:
                image_index[cid] = {"row_idx": row_idx, "captions": []}
            image_index[cid]["captions"].append(cap)

        # Ordered list for integer indexing
        self._entries = list(image_index.values())
        print(f"  Indexed {len(hf_dataset)} rows into {len(self._entries)} unique images")

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        entry = self._entries[idx]
        # Decode one image on demand
        image = self.hf_dataset[entry["row_idx"]]["image"]
        caption = random.choice(entry["captions"])

        encoded = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}
