from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools import mask as mask_utils

from transformers_compat import patch_optional_sklearn

patch_optional_sklearn()
from transformers import AutoTokenizer


class All_Dataset(data.Dataset):
    """RGB/SAR referring-segmentation dataset in RemoteSAM reference format."""

    def __init__(
        self,
        args,
        image_transforms=None,
        max_tokens: int = 20,
        split: str = "train",
        eval_mode: bool = True,
        logger=None,
    ) -> None:
        self.image_transforms = image_transforms
        self.split = split
        self.eval_mode = eval_mode
        self.max_tokens = max_tokens
        self.image_dir = Path(args.imageFolder).expanduser()
        self.ref_file = Path(args.ref_file_path).expanduser()
        self.instances_path = Path(args.instances_path).expanduser()
        text_encoder = getattr(args, "text_encoder", "modernbert")
        tokenizer_path = args.modernbert_path if text_encoder == "modernbert" else args.bert_tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

        self.samples = self._load_samples()
        self.annotations = self._load_annotations()
        self._validate_samples()

        if logger is not None:
            logger.info("Loaded %d %s samples from %s", len(self.samples), split, self.ref_file)

    def _load_samples(self) -> list[dict[str, Any]]:
        with self.ref_file.open("rb") as handle:
            refs = pickle.load(handle)

        samples: list[dict[str, Any]] = []
        for ref in refs:
            if ref.get("split") != self.split:
                continue
            ann_id = ref["ann_id"]
            sentence = ref["sentences"][0]["sent"]
            samples.append(
                {
                    "ann_id": ann_id,
                    "image_name": ref["file_name"],
                    "sentence": sentence,
                    "category_id": ref.get("category_id"),
                    "modal": ref.get("modal", "rgb"),
                }
            )
        return samples

    def _load_annotations(self) -> dict[int, dict[str, Any]]:
        with self.instances_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)
        return {ann["id"]: ann for ann in coco.get("annotations", [])}

    def _validate_samples(self) -> None:
        missing = [sample["ann_id"] for sample in self.samples if sample["ann_id"] not in self.annotations]
        if missing:
            preview = ", ".join(str(item) for item in missing[:5])
            raise KeyError(f"{len(missing)} refs have no matching COCO annotation: {preview}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, image_name: str) -> Path:
        image_path = self.image_dir / image_name
        if image_path.is_file():
            return image_path

        suffix_map = {".png": ".jpg", ".jpg": ".png", ".jpeg": ".png"}
        suffix = image_path.suffix.lower()
        if suffix in suffix_map:
            alternate = image_path.with_suffix(suffix_map[suffix])
            if alternate.is_file():
                return alternate
        raise FileNotFoundError(f"Image not found: {image_path}")

    @staticmethod
    def _decode_mask(segmentation: Any) -> np.ndarray:
        decoded = mask_utils.decode(segmentation)
        if decoded.ndim == 3:
            decoded = np.sum(decoded, axis=2)
        ref_mask = decoded.astype(np.uint8)
        target = np.zeros(ref_mask.shape, dtype=np.uint8)
        target[ref_mask == 1] = 1
        return target

    def _encode_sentence(self, sentence: str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self.max_tokens,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    @staticmethod
    def _modal_to_id(modal: Any) -> int:
        if isinstance(modal, (int, np.integer)):
            return int(modal)
        if isinstance(modal, torch.Tensor):
            return int(modal.reshape(-1)[0].item())
        return 1 if "sar" in str(modal).lower() else 0

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(self._resolve_image_path(sample["image_name"])).convert("RGB")

        annotation = self.annotations[sample["ann_id"]]
        ref_mask = self._decode_mask(annotation["segmentation"])
        target = Image.fromarray(ref_mask, mode="P")

        input_ids, attention_mask = self._encode_sentence(sample["sentence"])
        exist = torch.tensor([True], dtype=torch.bool)
        modal = torch.tensor(self._modal_to_id(sample.get("modal", "rgb")), dtype=torch.long)

        if self.image_transforms is not None:
            image, target = self.image_transforms(image, target)

        return image, target, input_ids, attention_mask, exist, modal
