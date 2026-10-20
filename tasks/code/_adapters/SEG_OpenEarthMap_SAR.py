from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import transforms as T
from tasks.code.runtime import load_model_and_text_encoder
from lib.transformers_compat import patch_optional_sklearn

patch_optional_sklearn()
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")


# Class order follows the existing SegEarthOV2/OpenEarthMap_SAR definition in
# /home/zhoujunjie/zhoujunjie/RemoteSAM-SAR/merge_data.py.
OPENEARTHMAP_SAR_CLASSES = [
    "background",
    "bareland",
    "grass",
    "pavement",
    "road",
    "tree",
    "water",
    "cropland",
    "building",
]
FOREGROUND_CLASSES = OPENEARTHMAP_SAR_CLASSES[1:]
FOREGROUND_CLASS_IDS = list(range(1, len(OPENEARTHMAP_SAR_CLASSES)))
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG", ".TIF", ".TIFF")
IGNORE_INDEX = 255


def resolve_image_path(directory: str | Path, stem: str) -> Path:
    directory = Path(directory)
    for suffix in IMAGE_SUFFIXES:
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Image not found for stem '{stem}' under {directory}")


def list_label_files(label_dir: str | Path) -> list[Path]:
    label_dir = Path(label_dir)
    files = [path for path in label_dir.iterdir() if path.is_file() and path.suffix in IMAGE_SUFFIXES]
    return sorted(files, key=lambda path: path.stem)


def get_transform(args):
    transforms = [
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(transforms)


def build_text_prompts(class_names: list[str], tokenizer, max_tokens: int):
    input_ids = {}
    attention_masks = {}
    for class_name in class_names:
        prompt_name = class_name.replace("_", " ")
        encoded = tokenizer(
            f"{prompt_name} in the image.",
            add_special_tokens=True,
            max_length=max_tokens,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids[class_name] = encoded["input_ids"]
        attention_masks[class_name] = encoded["attention_mask"]
    return input_ids, attention_masks


def forward_class(model, bert_model, image_tensor, input_ids, attention_mask, modal):
    if bert_model is not None:
        hidden_states = bert_model(input_ids, attention_mask=attention_mask)[0]
        embedding = hidden_states.permute(0, 2, 1).to(dtype=image_tensor.dtype)
        output = model(image_tensor, embedding, l_mask=attention_mask.unsqueeze(-1), modal=modal)
    else:
        output = model(image_tensor, input_ids, l_mask=attention_mask, modal=modal)
    return output


def load_label(label_path: Path, img_size: int, num_classes: int) -> tuple[np.ndarray, int, int]:
    label = Image.open(label_path)
    label = TF.resize(label, (img_size, img_size), interpolation=InterpolationMode.NEAREST)
    label_arr = np.array(label)
    if label_arr.ndim == 3:
        label_arr = label_arr[..., 0]
    label_arr = label_arr.astype(np.int64)

    mapped = np.full(label_arr.shape, IGNORE_INDEX, dtype=np.uint8)
    background_pixels = int((label_arr == 0).sum())
    valid = (label_arr > 0) & (label_arr < num_classes)
    mapped[valid] = label_arr[valid].astype(np.uint8)
    unknown_pixels = int(((label_arr < 0) | (label_arr >= num_classes)).sum())
    return mapped, unknown_pixels, background_pixels


def fast_hist(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    valid = (gt >= 0) & (gt < num_classes)
    return np.bincount(
        num_classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def per_class_iu(hist: np.ndarray) -> np.ndarray:
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_pa(hist: np.ndarray) -> np.ndarray:
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def evaluate(args):
    cudnn.benchmark = True
    cudnn.deterministic = True

    image_dir = Path(args.imageFolder)
    label_dir = Path(args.annoFolder)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"imageFolder does not exist: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"annoFolder does not exist: {label_dir}")

    label_files = list_label_files(label_dir)
    if not label_files:
        raise FileNotFoundError(f"No label images found under {label_dir}")

    model, bert_model, device = load_model_and_text_encoder(args, distributed=False)
    model.eval()
    if bert_model is not None:
        bert_model.eval()

    tokenizer_path = args.modernbert_path if getattr(args, "text_encoder", "modernbert") == "modernbert" else args.bert_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    prompt_ids, prompt_attn = build_text_prompts(FOREGROUND_CLASSES, tokenizer, max_tokens=args.max_tokens)
    transform = get_transform(args)

    num_classes = len(OPENEARTHMAP_SAR_CLASSES)
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    skipped = []
    unknown_pixels = 0
    ignored_background_pixels = 0
    total_pixels = 0
    modal = torch.tensor([1 if getattr(args, "modal", "rgb") == "sar" else 0], device=device, dtype=torch.long)

    print("Num classes", len(FOREGROUND_CLASSES))
    print("Ignored class: 0:background")
    print("Class order:", ", ".join(f"{idx}:{OPENEARTHMAP_SAR_CLASSES[idx]}" for idx in FOREGROUND_CLASS_IDS))
    print(f"Evaluating OpenEarthMap-SAR with modal={getattr(args, 'modal', 'rgb')}")

    with torch.no_grad():
        for label_path in tqdm(label_files, desc="Evaluating OpenEarthMap-SAR"):
            try:
                image_path = resolve_image_path(image_dir, label_path.stem)
            except FileNotFoundError:
                skipped.append(label_path.name)
                continue

            image = Image.open(image_path).convert("RGB")
            gt, unknown, background_pixels = load_label(label_path, args.img_size, num_classes)
            unknown_pixels += unknown
            ignored_background_pixels += background_pixels
            total_pixels += int(gt.size)

            image_tensor, _ = transform(image, image)
            image_tensor = image_tensor.unsqueeze(0).to(device)

            foreground_probs = []
            for class_name in FOREGROUND_CLASSES:
                ids = prompt_ids[class_name].to(device)
                attn = prompt_attn[class_name].to(device)
                output = forward_class(model, bert_model, image_tensor, ids, attn, modal)
                output = F.interpolate(output, (args.img_size, args.img_size), align_corners=True, mode="bilinear")
                foreground_probs.append(torch.softmax(output, dim=1)[:, 1].squeeze(0))

            pred_tensor = torch.stack(foreground_probs, dim=0).argmax(dim=0).to(torch.uint8) + 1
            pred = pred_tensor.cpu().numpy().astype(np.uint8)
            valid = gt != IGNORE_INDEX
            hist += fast_hist(gt[valid], pred[valid], num_classes)

    mious = per_class_iu(hist)
    mpas = per_class_pa(hist)
    mean_miou = float(np.nanmean(mious[FOREGROUND_CLASS_IDS]) * 100.0)
    mean_mpa = float(np.nanmean(mpas[FOREGROUND_CLASS_IDS]) * 100.0)
    mean_miou_all = float(np.nanmean(mious) * 100.0)
    mean_mpa_all = float(np.nanmean(mpas) * 100.0)

    print("OpenEarthMap-SAR semantic segmentation results:")
    for class_id in FOREGROUND_CLASS_IDS:
        class_name = OPENEARTHMAP_SAR_CLASSES[class_id]
        print(f"===>{class_name}:\tmIou-{round(mious[class_id] * 100, 2)}; mPA-{round(mpas[class_id] * 100, 2)}")
    print("Mean metrics are computed over the 8 foreground classes; background pixels are ignored.")
    print(f"===> mIoU: {round(mean_miou, 2)}; mPA: {round(mean_mpa, 2)}")
    print(f"===> mIoU_all_classes: {round(mean_miou_all, 2)}; mPA_all_classes: {round(mean_mpa_all, 2)}")

    if skipped:
        print(f"Skipped {len(skipped)} labels with no matching image. Examples: {skipped[:5]}")
    if ignored_background_pixels:
        ratio = ignored_background_pixels * 100.0 / max(total_pixels, 1)
        print(f"Ignored background-label pixels: {ignored_background_pixels} ({ratio:.4f}%)")
    if unknown_pixels:
        ratio = unknown_pixels * 100.0 / max(total_pixels, 1)
        print(f"Ignored unknown-label pixels: {unknown_pixels} ({ratio:.4f}%)")

    metrics = {
        "mIoU": mean_miou,
        "mPA": mean_mpa,
        "mIoU_all_classes": mean_miou_all,
        "mPA_all_classes": mean_mpa_all,
        "mean_note": "8 foreground classes only; ground-truth background pixels are ignored",
        "classes": {
            class_name: {
                "id": class_id,
                "mIoU": float(mious[class_id] * 100.0),
                "mPA": float(mpas[class_id] * 100.0),
            }
            for class_id, class_name in enumerate(OPENEARTHMAP_SAR_CLASSES)
        },
        "skipped": skipped,
        "unknown_pixels": unknown_pixels,
        "ignored_background_pixels": ignored_background_pixels,
        "total_pixels": total_pixels,
        "modal": getattr(args, "modal", "rgb"),
        "imageFolder": str(image_dir),
        "annoFolder": str(label_dir),
    }

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        output_path = os.path.join(args.save_path, "seg_openearthmap_sar_results.json")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"Results saved to {output_path}")

    return metrics


def main(args):
    evaluate(args)


if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    args = parser.parse_args()
    main(args)
