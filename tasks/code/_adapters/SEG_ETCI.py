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
from transformers_compat import patch_optional_sklearn

patch_optional_sklearn()
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")


ETCI_CLASSES = ["background", "flood"]
IMAGE_SUFFIXES = (".png", ".PNG")
IGNORE_INDEX = 255


def get_transform(args):
    transforms = [
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(transforms)


def build_text_prompt(class_name: str, tokenizer, max_tokens: int):
    encoded = tokenizer(
        f"{class_name.replace('_', ' ')} in the image.",
        add_special_tokens=True,
        max_length=max_tokens,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def forward_binary(model, bert_model, image_tensor, input_ids, attention_mask, modal):
    if bert_model is not None:
        hidden_states = bert_model(input_ids, attention_mask=attention_mask)[0]
        embedding = hidden_states.permute(0, 2, 1).to(dtype=image_tensor.dtype)
        output = model(image_tensor, embedding, l_mask=attention_mask.unsqueeze(-1), modal=modal)
    else:
        output = model(image_tensor, input_ids, l_mask=attention_mask, modal=modal)
    return output


def list_label_files(etci_root: str | Path, label_name: str) -> list[Path]:
    root = Path(etci_root)
    files = []
    for event_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label_dir = event_dir / "tiles" / label_name
        if not label_dir.is_dir():
            continue
        files.extend(sorted(path for path in label_dir.iterdir() if path.is_file() and path.suffix in IMAGE_SUFFIXES))
    return files


def image_path_for_label(label_path: Path, polarization: str) -> Path:
    tiles_dir = label_path.parents[1]
    image_name = f"{label_path.stem}_{polarization}.png"
    image_path = tiles_dir / polarization / image_name
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found for label {label_path}: expected {image_path}")
    return image_path


def load_binary_label(label_path: Path, img_size: int) -> tuple[np.ndarray, int]:
    label = Image.open(label_path)
    label = TF.resize(label, (img_size, img_size), interpolation=InterpolationMode.NEAREST)
    label_arr = np.array(label)
    if label_arr.ndim == 3:
        label_arr = label_arr[..., 0]

    mapped = np.full(label_arr.shape, IGNORE_INDEX, dtype=np.uint8)
    mapped[label_arr == 0] = 0
    mapped[label_arr > 0] = 1
    unknown_pixels = int((mapped == IGNORE_INDEX).sum())
    return mapped, unknown_pixels


def fast_hist(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    valid = (gt >= 0) & (gt < num_classes)
    return np.bincount(
        num_classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def compute_metrics(hist: np.ndarray) -> dict:
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
        "mF1": float(f1[1] * 100.0),
        "mIoU": float(iou[1] * 100.0),
        "mF1_all_classes": float(np.nanmean(f1) * 100.0),
        "mIoU_all_classes": float(np.nanmean(iou) * 100.0),
        "pixel_acc": float(pixel_acc * 100.0),
    }


def evaluate(args):
    cudnn.benchmark = True
    cudnn.deterministic = True

    etci_root = Path(args.etci_root)
    if not etci_root.is_dir():
        raise FileNotFoundError(f"ETCI test root does not exist: {etci_root}")
    if args.etci_label not in {"flood_label", "water_body_label"}:
        raise ValueError("--etci_label must be 'flood_label' or 'water_body_label'")
    if args.etci_polarization not in {"vv", "vh"}:
        raise ValueError("--etci_polarization must be 'vv' or 'vh'")

    label_files = list_label_files(etci_root, args.etci_label)
    if args.max_samples and args.max_samples > 0:
        label_files = label_files[: args.max_samples]
    if not label_files:
        raise FileNotFoundError(f"No {args.etci_label} label images found under {etci_root}")

    model, bert_model, device = load_model_and_text_encoder(args, distributed=False)
    model.eval()
    if bert_model is not None:
        bert_model.eval()

    tokenizer_path = args.modernbert_path if getattr(args, "text_encoder", "modernbert") == "modernbert" else args.bert_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    prompt_name = args.etci_prompt
    input_ids, attention_mask = build_text_prompt(prompt_name, tokenizer, max_tokens=args.max_tokens)
    transform = get_transform(args)

    hist = np.zeros((len(ETCI_CLASSES), len(ETCI_CLASSES)), dtype=np.int64)
    skipped = []
    unknown_pixels = 0
    total_pixels = 0
    modal = torch.tensor([1 if getattr(args, "modal", "rgb") == "sar" else 0], device=device, dtype=torch.long)

    print("ETCI 2021 Flood Detection semantic segmentation results")
    print(f"Root: {etci_root}")
    print(f"Label: {args.etci_label}")
    print(f"Polarization: {args.etci_polarization}")
    print(f"Prompt: {prompt_name}")
    print(f"Samples: {len(label_files)}")

    with torch.no_grad():
        for label_path in tqdm(label_files, desc="Evaluating ETCI"):
            try:
                image_path = image_path_for_label(label_path, args.etci_polarization)
            except FileNotFoundError:
                skipped.append(str(label_path))
                continue

            image = Image.open(image_path).convert("RGB")
            gt, unknown = load_binary_label(label_path, args.img_size)
            unknown_pixels += unknown
            total_pixels += int(gt.size)

            image_tensor, _ = transform(image, image)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            ids = input_ids.to(device)
            attn = attention_mask.to(device)
            output = forward_binary(model, bert_model, image_tensor, ids, attn, modal)
            output = F.interpolate(output, (args.img_size, args.img_size), align_corners=True, mode="bilinear")
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            valid = gt != IGNORE_INDEX
            hist += fast_hist(gt[valid], pred[valid], len(ETCI_CLASSES))

    metrics = compute_metrics(hist)
    result_rows = {}

    print("Class\tPrecision\tRecall\tF1\tIoU")
    for index, class_name in enumerate(ETCI_CLASSES):
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

    print("Mean metrics are reported on the foreground class.")
    print(f"===> mF1: {metrics['mF1']:.2f}")
    print(f"===> mIoU: {metrics['mIoU']:.2f}")
    print(f"===> mF1_all_classes: {metrics['mF1_all_classes']:.2f}")
    print(f"===> mIoU_all_classes: {metrics['mIoU_all_classes']:.2f}")
    print(f"===> pixel_acc: {metrics['pixel_acc']:.2f}")
    if skipped:
        print(f"Skipped {len(skipped)} labels with no matching image. Examples: {skipped[:5]}")
    if unknown_pixels:
        ratio = unknown_pixels * 100.0 / max(total_pixels, 1)
        print(f"Ignored unknown-label pixels: {unknown_pixels} ({ratio:.4f}%)")

    output = {
        "mF1": metrics["mF1"],
        "mIoU": metrics["mIoU"],
        "mF1_all_classes": metrics["mF1_all_classes"],
        "mIoU_all_classes": metrics["mIoU_all_classes"],
        "pixel_acc": metrics["pixel_acc"],
        "mean_note": "foreground class only",
        "classes": result_rows,
        "hist": hist.tolist(),
        "skipped": skipped,
        "unknown_pixels": unknown_pixels,
        "total_pixels": total_pixels,
        "etci_root": str(etci_root),
        "etci_label": args.etci_label,
        "etci_polarization": args.etci_polarization,
        "etci_prompt": prompt_name,
        "modal": getattr(args, "modal", "rgb"),
        "num_samples": len(label_files),
    }

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        output_path = os.path.join(
            args.save_path,
            f"seg_etci_{args.etci_label}_{args.etci_polarization}_results.json",
        )
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
        print(f"Results saved to {output_path}")

    return output


def main(args):
    evaluate(args)


if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    parser.add_argument(
        "--etci_root",
        default="/data/user2/zjj/sar_downstream_datasets/ETCI-2021-Flood-Detection/data/test",
        help="Path to the ETCI data/test directory.",
    )
    parser.add_argument(
        "--etci_label",
        default="flood_label",
        choices=["flood_label", "water_body_label"],
        help="Binary ETCI label directory to evaluate.",
    )
    parser.add_argument(
        "--etci_polarization",
        default="vh",
        choices=["vv", "vh"],
        help="SAR polarization image directory to use.",
    )
    parser.add_argument(
        "--etci_prompt",
        default="flooded water",
        help="Text prompt class for binary foreground segmentation.",
    )
    parser.add_argument(
        "--max_samples",
        default=0,
        type=int,
        help="Optional debug limit; 0 evaluates all ETCI test tiles.",
    )
    args = parser.parse_args()
    main(args)
