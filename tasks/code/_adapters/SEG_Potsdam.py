from __future__ import annotations

import json
import os
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import transforms as T
from tasks.code.runtime import load_model_and_text_encoder
from transformers_compat import patch_optional_sklearn

patch_optional_sklearn()
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")


POTSDAM_CLASSES = OrderedDict(
    [
        ("impervious_surface", (255, 255, 255)),
        ("building", (0, 0, 255)),
        ("low_vegetation", (0, 255, 255)),
        ("tree", (0, 255, 0)),
        ("car", (255, 255, 0)),
        ("clutter", (255, 0, 0)),
    ]
)

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


def rgb_to_label_map(label_rgb: Image.Image, class_palette: OrderedDict[str, tuple[int, int, int]]) -> np.ndarray:
    rgb = np.array(label_rgb.convert("RGB"), dtype=np.uint8)
    label = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
    for class_id, color in enumerate(class_palette.values()):
        label[np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)] = class_id
    return label


def fast_hist(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    valid = (gt >= 0) & (gt < num_classes)
    return np.bincount(
        num_classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def compute_metrics(hist: np.ndarray, mean_indices: list[int]) -> dict:
    tp = np.diag(hist).astype(np.float64)
    gt_count = hist.sum(axis=1).astype(np.float64)
    pred_count = hist.sum(axis=0).astype(np.float64)
    fp = pred_count - tp
    fn = gt_count - tp

    precision = np.divide(tp, tp + fp, out=np.full_like(tp, np.nan), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan), where=(tp + fn) > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.full_like(tp, np.nan), where=(2 * tp + fp + fn) > 0)
    iou = np.divide(tp, tp + fp + fn, out=np.full_like(tp, np.nan), where=(tp + fp + fn) > 0)
    pixel_acc = tp.sum() / max(hist.sum(), 1)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "mF1": float(np.nanmean(f1[mean_indices]) * 100.0),
        "mIoU": float(np.nanmean(iou[mean_indices]) * 100.0),
        "mF1_all": float(np.nanmean(f1) * 100.0),
        "mIoU_all": float(np.nanmean(iou) * 100.0),
        "pixel_acc": float(pixel_acc * 100.0),
    }


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
    return torch.softmax(output, dim=1)[:, 1]


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
    class_names = list(POTSDAM_CLASSES.keys())
    prompt_ids, prompt_attn = build_text_prompts(class_names, tokenizer, max_tokens=args.max_tokens)
    transform = get_transform(args)

    hist = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    unknown_pixels = 0
    total_pixels = 0
    skipped = []

    modal = torch.tensor([1 if getattr(args, "modal", "rgb") == "sar" else 0], device=device, dtype=torch.long)

    with torch.no_grad():
        for label_path in tqdm(label_files, desc="Evaluating Potsdam"):
            try:
                image_path = resolve_image_path(image_dir, label_path.stem)
            except FileNotFoundError:
                skipped.append(label_path.name)
                continue

            image = Image.open(image_path).convert("RGB")
            label_rgb = Image.open(label_path).convert("RGB")
            label_rgb = TF.resize(label_rgb, (args.img_size, args.img_size), interpolation=InterpolationMode.NEAREST)
            gt = rgb_to_label_map(label_rgb, POTSDAM_CLASSES)
            unknown_pixels += int((gt == IGNORE_INDEX).sum())
            total_pixels += int(gt.size)

            image_tensor, _ = transform(image, image)
            image_tensor = image_tensor.unsqueeze(0).to(device)

            prob_maps = []
            for class_name in class_names:
                ids = prompt_ids[class_name].to(device)
                attn = prompt_attn[class_name].to(device)
                prob_maps.append(forward_class(model, bert_model, image_tensor, ids, attn, modal).squeeze(0))

            probs = torch.stack(prob_maps, dim=0)
            pred = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)
            valid = gt != IGNORE_INDEX
            hist += fast_hist(gt[valid], pred[valid], len(class_names))

    if getattr(args, "include_clutter_in_mean", False):
        mean_indices = list(range(len(class_names)))
        mean_note = "including clutter"
    else:
        mean_indices = [index for index, name in enumerate(class_names) if name != "clutter"]
        mean_note = "excluding clutter"
    metrics = compute_metrics(hist, mean_indices)

    print("Potsdam semantic segmentation results:")
    print("Class\tPrecision\tRecall\tF1\tIoU")
    result_rows = {}
    for index, class_name in enumerate(class_names):
        precision = metrics["precision"][index] * 100.0
        recall = metrics["recall"][index] * 100.0
        f1 = metrics["f1"][index] * 100.0
        iou = metrics["iou"][index] * 100.0
        print(f"{class_name}\t{precision:.2f}\t{recall:.2f}\t{f1:.2f}\t{iou:.2f}")
        result_rows[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
        }

    print(f"Mean metrics are computed over {mean_note}.")
    print(f"===> mF1: {metrics['mF1']:.2f}")
    print(f"===> mIoU: {metrics['mIoU']:.2f}")
    print(f"===> mF1_all_classes: {metrics['mF1_all']:.2f}")
    print(f"===> mIoU_all_classes: {metrics['mIoU_all']:.2f}")
    print(f"===> pixel_acc: {metrics['pixel_acc']:.2f}")
    if skipped:
        print(f"Skipped {len(skipped)} labels with no matching image. Examples: {skipped[:5]}")
    if unknown_pixels:
        ratio = unknown_pixels * 100.0 / max(total_pixels, 1)
        print(f"Ignored unknown-label pixels: {unknown_pixels} ({ratio:.4f}%)")

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        output_path = os.path.join(args.save_path, "seg_potsdam_results.json")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "mF1": metrics["mF1"],
                    "mIoU": metrics["mIoU"],
                    "mF1_all_classes": metrics["mF1_all"],
                    "mIoU_all_classes": metrics["mIoU_all"],
                    "mean_note": mean_note,
                    "pixel_acc": metrics["pixel_acc"],
                    "classes": result_rows,
                    "skipped": skipped,
                    "unknown_pixels": unknown_pixels,
                    "total_pixels": total_pixels,
                },
                handle,
                indent=2,
            )
        print(f"Results saved to {output_path}")

    return metrics


def main(args):
    evaluate(args)


if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    parser.add_argument(
        "--include_clutter_in_mean",
        action="store_true",
        help="Include the Potsdam clutter/background class when computing mF1/mIoU.",
    )
    args = parser.parse_args()
    main(args)
