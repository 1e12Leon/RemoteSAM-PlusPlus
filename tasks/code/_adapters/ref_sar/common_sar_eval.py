from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset, SequentialSampler

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils
from args import get_parser
from lib import segmentation
from lib.text_encoders import build_text_encoder_model
from lib.transformers_compat import patch_optional_sklearn

patch_optional_sklearn()
from transformers import AutoTokenizer

from tasks.code._adapters import ref_utils as _TEST_UTILS

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_CHECKPOINT = ROOT / "checkpoints/model_best_mIoU_RemoteSAM_DINOv3ConvNeXtB_ModernBERT.pth"
DEFAULT_EXTRACTED_ROOT = Path("/data/user2/RemoteSAM_SAR_Source520K_BySplit")
IGNORE_VALUES = {255}


@dataclass(frozen=True)
class SarRecord:
    image_path: Path
    mask_path: Path
    class_name: str
    mask_value: int | None
    prompt: str


class RawSarDataset(Dataset):
    def __init__(self, records: Sequence[SarRecord], args) -> None:
        self.records = list(records)
        self.args = args
        self.transform = _TEST_UTILS.get_transform(args)
        tokenizer_path = args.modernbert_path if args.text_encoder == "modernbert" else args.bert_tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        mask = read_mask(record.mask_path)
        if record.mask_value is None:
            target_arr = (mask > 0).astype(np.uint8)
        else:
            target_arr = (mask == record.mask_value).astype(np.uint8)
        target = Image.fromarray(target_arr, mode="P")
        image, target = self.transform(image, target)

        encoded = self.tokenizer(
            record.prompt,
            add_special_tokens=True,
            max_length=self.args.max_tokens,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        exist = torch.tensor([True], dtype=torch.bool)
        modal = torch.tensor(1, dtype=torch.long)
        return image, target, encoded["input_ids"], encoded["attention_mask"], exist, modal, record.class_name


def infer_class_name(sentence: str, fallback: str) -> str:
    sentence = str(sentence).strip()
    match = re.match(r"^A region of (.+?)(?: at | in | on | distributed|\\.|$)", sentence, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    match = re.match(r"^(.+?) in the (?:SAR )?image\\.?$", sentence, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return fallback


class RefSarDataset(Dataset):
    """RemoteSAM refs.p + instances.json SAR evaluation dataset."""

    def __init__(self, data_root: Path, split: str, args) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.args = args
        self.image_dir = self.data_root / split / "images"
        self.ref_file = self.data_root / split / "refs.p"
        self.instances_path = self.data_root / split / "instances.json"
        if not self.ref_file.is_file():
            raise FileNotFoundError(f"Missing refs.p: {self.ref_file}")
        if not self.instances_path.is_file():
            raise FileNotFoundError(f"Missing instances.json: {self.instances_path}")

        with self.ref_file.open("rb") as handle:
            refs = pickle.load(handle)
        with self.instances_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)

        self.annotations = {ann["id"]: ann for ann in coco.get("annotations", [])}
        self.category_names = {
            int(cat["id"]): str(cat.get("name", cat["id"])) for cat in coco.get("categories", [])
        }
        self.samples = []
        for ref in refs:
            if ref.get("split") != split:
                continue
            ann_id = ref["ann_id"]
            if ann_id not in self.annotations:
                continue
            sentence = ref["sentences"][0].get("sent") or ref["sentences"][0].get("raw", "")
            ann = self.annotations[ann_id]
            category_id = ref.get("category_id", ann.get("category_id", ann.get("categories_id")))
            fallback = self.category_names.get(int(category_id), str(category_id)) if category_id is not None else "unknown"
            class_name = infer_class_name(sentence, fallback)
            self.samples.append(
                {
                    "ann_id": ann_id,
                    "file_name": ref["file_name"],
                    "sentence": sentence,
                    "class_name": class_name,
                }
            )

        if args.limit and args.limit > 0:
            self.samples = self.samples[: args.limit]

        self.transform = _TEST_UTILS.get_transform(args)
        tokenizer_path = args.modernbert_path if args.text_encoder == "modernbert" else args.bert_tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, file_name: str) -> Path:
        image_path = self.image_dir / file_name
        if image_path.is_file() or image_path.is_symlink():
            return image_path
        suffix_map = {".png": ".jpg", ".jpg": ".png", ".jpeg": ".png", ".tif": ".png", ".tiff": ".png"}
        alternate_suffix = suffix_map.get(image_path.suffix.lower())
        if alternate_suffix:
            alternate = image_path.with_suffix(alternate_suffix)
            if alternate.is_file() or alternate.is_symlink():
                return alternate
        raise FileNotFoundError(f"Image not found: {image_path}")

    @staticmethod
    def _decode_mask(segmentation) -> np.ndarray:
        decoded = mask_utils.decode(segmentation)
        if decoded.ndim == 3:
            decoded = np.sum(decoded, axis=2)
        target = np.zeros(decoded.shape, dtype=np.uint8)
        target[decoded > 0] = 1
        return target

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(self._resolve_image_path(sample["file_name"])).convert("RGB")
        ann = self.annotations[sample["ann_id"]]
        target = Image.fromarray(self._decode_mask(ann["segmentation"]), mode="P")
        image, target = self.transform(image, target)

        encoded = self.tokenizer(
            sample["sentence"],
            add_special_tokens=True,
            max_length=self.args.max_tokens,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        exist = torch.tensor([True], dtype=torch.bool)
        modal = torch.tensor(1, dtype=torch.long)
        return image, target, encoded["input_ids"], encoded["attention_mask"], exist, modal, sample["class_name"]


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = get_parser()
    parser.description = description
    parser.set_defaults(
        resume=str(DEFAULT_CHECKPOINT),
        img_size=896,
        modernbert_path=str(ROOT / "gte-modernbert-base"),
        dinov3_convnext_path=str(ROOT / "dinov3-convnext-B"),
        modal="sar",
    )
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--prompt_template", default=None)
    parser.add_argument("--min_area", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def read_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def iter_images(image_dir: Path) -> Iterable[Path]:
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS:
            yield path


def find_by_stem(directory: Path, stem_or_name: str) -> Path | None:
    candidate = directory / stem_or_name
    if candidate.is_file():
        return candidate
    stem = Path(stem_or_name).stem
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def read_mask(mask_path: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def describe_location(mask: np.ndarray, mask_value: int | None) -> str:
    if mask_value is None:
        foreground = mask > 0
    else:
        foreground = mask == mask_value
    ys, xs = np.nonzero(foreground)
    if len(xs) == 0:
        return "in the center of the image"

    y_pos = float(ys.mean()) / max(1, mask.shape[0] - 1)
    x_pos = float(xs.mean()) / max(1, mask.shape[1] - 1)
    vertical = "top" if y_pos < 0.35 else "bottom" if y_pos > 0.65 else ""
    horizontal = "left" if x_pos < 0.35 else "right" if x_pos > 0.65 else ""
    if vertical and horizontal:
        return f"at the {vertical}-{horizontal}"
    if vertical:
        return f"at the {vertical}"
    if horizontal:
        return f"on the {horizontal}"
    return "in the center of the image"


def default_prompt(class_name: str, location: str) -> str:
    normalized = class_name.lower().replace("_", " ").replace("-", " ")
    road_like = {"road", "highway", "railway", "runway"}
    built_like = {
        "building",
        "buildings",
        "house",
        "city",
        "residential",
        "industrial",
        "airport",
        "harbor",
        "bridge",
    }
    natural_like = {
        "vegetation",
        "greenery",
        "grass",
        "tree",
        "trees",
        "forest",
        "farmland",
        "cropland",
        "bareland",
        "bare ground",
        "bareground",
        "pavement",
        "water",
        "river",
        "lake",
        "sea",
    }

    if normalized in road_like:
        second = "It is extending across the terrain."
    elif normalized in built_like:
        second = "It is standing on the ground."
    elif normalized in natural_like:
        second = "It is visible in the scene."
    else:
        second = "It is visible in the SAR scene."
    return f"A region of {class_name} {location}. {second}"


def make_prompt(template: str | None, class_name: str, mask: np.ndarray | None = None, mask_value: int | None = None) -> str:
    if template:
        return template.format(class_name=class_name)
    if mask is None:
        location = "in the center of the image"
    else:
        location = describe_location(mask, mask_value)
    return default_prompt(class_name, location)


def semantic_records(
    pairs: Iterable[tuple[Path, Path]],
    class_names: Sequence[str],
    prompt_template: str,
    min_area: int,
) -> list[SarRecord]:
    records: list[SarRecord] = []
    for image_path, mask_path in pairs:
        mask = read_mask(mask_path)
        for raw_value in np.unique(mask).tolist():
            value = int(raw_value)
            if value in IGNORE_VALUES or value < 0 or value >= len(class_names):
                continue
            class_name = class_names[value]
            if class_name.lower() == "background":
                continue
            if int((mask == value).sum()) < min_area:
                continue
            records.append(
                SarRecord(
                    image_path=image_path,
                    mask_path=mask_path,
                    class_name=class_name,
                    mask_value=value,
                    prompt=make_prompt(prompt_template, class_name, mask, value),
                )
            )
    return records


def pairs_from_dirs(image_dir: Path, label_dir: Path, names: Sequence[str] | None = None) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    if names is None:
        image_paths = list(iter_images(image_dir))
    else:
        image_paths = []
        for name in names:
            image_path = find_by_stem(image_dir, name)
            if image_path is not None:
                image_paths.append(image_path)
    for image_path in image_paths:
        mask_path = find_by_stem(label_dir, image_path.name)
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    return pairs


def prepare_args(args) -> None:
    checkpoint = _TEST_UTILS.torch_load_checkpoint(args.resume, map_location="cpu")
    if _TEST_UTILS.checkpoint_has_modal_lora(checkpoint) and not args.use_modal_lora:
        args.use_modal_lora = True
        print("Detected modal LoRA weights in checkpoint; enabling --use_modal_lora.")
    _TEST_UTILS.maybe_set_window_size_from_checkpoint(args, checkpoint)
    args._loaded_checkpoint = checkpoint


def build_model(args, device: torch.device):
    checkpoint = getattr(args, "_loaded_checkpoint", None)
    if checkpoint is None:
        checkpoint = _TEST_UTILS.torch_load_checkpoint(args.resume, map_location="cpu")
    model = segmentation.__dict__[args.model](pretrained="", args=args)
    _TEST_UTILS.load_weights(model, checkpoint)
    model = model.to(device)

    if args.model == "lavt_one":
        return model, None

    bert_model = build_text_encoder_model(args)
    if "bert_model" in checkpoint:
        bert_state = checkpoint["bert_model"]
        if any(key.startswith("module.") for key in bert_state):
            bert_state = {_TEST_UTILS.remove_prefix(key, "module."): value for key, value in bert_state.items()}
        bert_model.load_state_dict(bert_state, strict=False)
    return model, bert_model.to(device)


def evaluate_ref_split(args, title: str) -> dict:
    prepare_args(args)
    dataset = RefSarDataset(args.data_root, args.split, args)
    if not dataset:
        raise RuntimeError(f"No SAR evaluation refs found for {title}/{args.split}.")
    return evaluate_dataset(args, dataset, f"{title}/{args.split}")


def evaluate_records(args, records: Sequence[SarRecord], title: str) -> dict:
    if args.limit and args.limit > 0:
        records = list(records)[: args.limit]
    if not records:
        raise RuntimeError(f"No SAR evaluation records found for {title}.")

    prepare_args(args)
    dataset = RawSarDataset(records, args)
    return evaluate_dataset(args, dataset, title)


@torch.no_grad()
def evaluate_dataset(args, dataset: Dataset, title: str) -> dict:
    device = torch.device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(dataset),
        num_workers=args.workers,
        pin_memory=args.pin_mem,
    )
    model, bert_model = build_model(args, device)
    model.eval()
    if bert_model is not None:
        bert_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    thresholds = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    precision_hits = np.zeros(len(thresholds), dtype=np.int64)
    mean_ious: list[float] = []
    by_class: dict[str, list[float]] = defaultdict(list)
    cum_i, cum_u = 0, 0

    for image, target, sentences, attentions, _, modal, class_names in metric_logger.log_every(
        loader, 100, f"Test {title}:"
    ):
        image = image.to(device, non_blocking=True)
        target_np = target.numpy()
        sentences = sentences.squeeze(1).to(device, non_blocking=True)
        attentions = attentions.squeeze(1).to(device, non_blocking=True)
        modal = _TEST_UTILS.modal_to_tensor(modal, image.device)

        if bert_model is None:
            output = model(image, sentences, attentions, modal=modal)
        else:
            hidden_states = bert_model(sentences, attention_mask=attentions)[0]
            embedding = hidden_states.permute(0, 2, 1).to(dtype=image.dtype)
            output = model(image, embedding, attentions.unsqueeze(-1), modal=modal)

        prediction = output.argmax(1).cpu().numpy()
        for index in range(prediction.shape[0]):
            intersection, union = _TEST_UTILS.compute_iou(prediction[index], target_np[index])
            this_iou = 0.0 if union == 0 else intersection / union
            mean_ious.append(this_iou)
            class_name = class_names[index] if isinstance(class_names, (list, tuple)) else str(class_names)
            by_class[class_name].append(this_iou)
            cum_i += intersection
            cum_u += union
            precision_hits += this_iou >= thresholds

    metrics = {
        "samples": len(mean_ious),
        "mIoU": float(np.mean(mean_ious) * 100.0),
        "oIoU": float(100.0 * cum_i / cum_u) if cum_u > 0 else 0.0,
        "precision": {
            str(threshold): float(hits * 100.0 / max(1, len(mean_ious)))
            for threshold, hits in zip(thresholds, precision_hits)
        },
        "per_class_mIoU": {
            class_name: float(np.mean(values) * 100.0) for class_name, values in sorted(by_class.items())
        },
    }

    print(f"Final {title} SAR results:")
    print(f"    samples = {metrics['samples']}")
    print("    Mean IoU = %.2f" % metrics["mIoU"])
    print("    overall IoU = %.2f" % metrics["oIoU"])
    for threshold, precision in metrics["precision"].items():
        print("    precision@%s = %.2f" % (threshold, precision))
    print("    per-class mIoU:")
    for class_name, score in metrics["per_class_mIoU"].items():
        print("        %s = %.2f" % (class_name, score))
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
    return metrics
