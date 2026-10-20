from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import transforms as T
from tasks.code.runtime import load_model_and_text_encoder
from transformers_compat import patch_optional_sklearn

patch_optional_sklearn()
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")


IGNORE_INDEX = 255
DEFAULT_SEG_SAR_ROOT = "/data/user2/SAR_Data/SegEarthOV2"
BACKGROUND_CLASS_NAMES = {"background"}
YESEG_LEGACY_PROMPT_CLASSES = (
    "bareground,barren",
    "grass,farmland",
    "dense tree cover",
    "city",
    "water",
    "roadway",
)
YESEG_CURRENT_PROMPT_CLASSES = ("bareground", "grass", "tree", "house", "water", "road")


@dataclass(frozen=True)
class SegSARDatasetConfig:
    key: str
    display_name: str
    classes: tuple[str, ...]
    image_dir: str
    label_dir: str
    image_suffix: str
    label_suffix: str
    eval_size: int
    ann_file: str | None = None
    reduce_zero_label: bool = False
    prompt_classes: tuple[str, ...] | None = None


# These configs mirror the SAR entries in SegEarth-OV-2/eval_all.py and
# configs/cfg_*_sar.py. prompt_classes mirrors configs/cls_*_sar.txt where
# SegEarth-OV-2 uses text aliases that differ from the label class names.
SEG_SAR_DATASETS: dict[str, SegSARDatasetConfig] = {
    "openearthmap_sar": SegSARDatasetConfig(
        key="openearthmap_sar",
        display_name="OpenEarthMap-SAR",
        classes=(
            "background",
            "bareland",
            "grass",
            "pavement",
            "road",
            "tree",
            "water",
            "cropland",
            "building",
        ),
        prompt_classes=(
            "background",
            "bareland,barren",
            "grass",
            "pavement",
            "road",
            "tree,forest",
            "water,river",
            "cropland",
            "building,roof,house",
        ),
        image_dir="OpenEarthMap_SAR/test/sar_images",
        label_dir="OpenEarthMap_SAR/test/labels",
        image_suffix=".tif",
        label_suffix=".tif",
        eval_size=448,
        reduce_zero_label=False,
    ),
    "ddhrnet_korea_sar": SegSARDatasetConfig(
        key="ddhrnet_korea_sar",
        display_name="DDHRNet-Korea-SAR",
        classes=("building", "road", "greenery", "water", "farmland"),
        prompt_classes=("building", "road", "greenery", "water", "farmland,grass"),
        image_dir="DDHRNet_DATA/korea/no_cloud/sar",
        label_dir="DDHRNet_DATA/korea/no_cloud/label",
        ann_file="DDHRNet_DATA/korea/no_cloud/val.txt",
        image_suffix=".jpg",
        label_suffix=".png",
        eval_size=224,
        reduce_zero_label=False,
    ),
    "ddhrnet_shandong_sar": SegSARDatasetConfig(
        key="ddhrnet_shandong_sar",
        display_name="DDHRNet-Shandong-SAR",
        classes=("farmland", "greenery", "road", "building", "water"),
        prompt_classes=("farmland,grass", "greenery", "road", "building", "water"),
        image_dir="DDHRNet_DATA/shandong/sar",
        label_dir="DDHRNet_DATA/shandong/label",
        ann_file="DDHRNet_DATA/shandong/val.txt",
        image_suffix=".jpg",
        label_suffix=".png",
        eval_size=224,
        reduce_zero_label=False,
    ),
    "ddhrnet_xian_sar": SegSARDatasetConfig(
        key="ddhrnet_xian_sar",
        display_name="DDHRNet-Xian-SAR",
        classes=("building", "road", "farmland", "greenery", "water"),
        prompt_classes=("building", "road", "farmland,grass", "greenery", "water"),
        image_dir="DDHRNet_DATA/xian/no_cloud/sar",
        label_dir="DDHRNet_DATA/xian/no_cloud/label",
        ann_file="DDHRNet_DATA/xian/no_cloud/val.txt",
        image_suffix=".jpg",
        label_suffix=".png",
        eval_size=224,
        reduce_zero_label=False,
    ),
    "fusar_sar": SegSARDatasetConfig(
        key="fusar_sar",
        display_name="FUSAR-Map",
        classes=("background", "water", "road", "building", "vegetation"),
        prompt_classes=("others", "water", "road", "building", "grass"),
        image_dir="FUSAR-Map/sar",
        label_dir="FUSAR-Map/label",
        ann_file="FUSAR-Map/test.txt",
        image_suffix=".tif",
        label_suffix=".png",
        eval_size=448,
        reduce_zero_label=False,
    ),
    "whu_sar": SegSARDatasetConfig(
        key="whu_sar",
        display_name="WHU-OPT-SAR",
        classes=("farmland", "city", "village", "water", "forest", "road", "background"),
        prompt_classes=("farmland", "city", "village", "water", "forest", "road", "others"),
        image_dir="whu-opt-sar/crop512/test/sar_dir",
        label_dir="whu-opt-sar/crop512/test/ann_dir",
        image_suffix=".tif",
        label_suffix=".tif",
        eval_size=224,
        reduce_zero_label=True,
    ),
    "yeseg_sar": SegSARDatasetConfig(
        key="yeseg_sar",
        display_name="YESeg-OPT-SAR",
        classes=("bareground", "grass", "tree", "house", "water", "road"),
        prompt_classes=YESEG_LEGACY_PROMPT_CLASSES,
        image_dir="YESeg-OPT-SAR/sar",
        label_dir="YESeg-OPT-SAR/cvt_label",
        ann_file="YESeg-OPT-SAR/test.txt",
        image_suffix=".png",
        label_suffix=".png",
        eval_size=224,
        reduce_zero_label=False,
    ),
    "pie_sar": SegSARDatasetConfig(
        key="pie_sar",
        display_name="PIE-RGB-SAR",
        classes=("background", "city", "road", "water", "forest", "farmland"),
        prompt_classes=("background", "city", "road", "water,river", "forest", "cropland"),
        image_dir="PIE-RGB-SAR/sar",
        label_dir="PIE-RGB-SAR/label",
        ann_file="PIE-RGB-SAR/val.txt",
        image_suffix=".tif",
        label_suffix=".tif",
        eval_size=224,
        reduce_zero_label=False,
    ),
}

DATASET_ALIASES = {
    "openearthmap": "openearthmap_sar",
    "openearthmap-sar": "openearthmap_sar",
    "ddhr_korea_sar": "ddhrnet_korea_sar",
    "ddhrnet-korea-sar": "ddhrnet_korea_sar",
    "ddhr_shandong_sar": "ddhrnet_shandong_sar",
    "ddhrnet-shandong-sar": "ddhrnet_shandong_sar",
    "ddhr_xian_sar": "ddhrnet_xian_sar",
    "ddhrnet-xian-sar": "ddhrnet_xian_sar",
    "fusar": "fusar_sar",
    "fusar-map": "fusar_sar",
    "whu": "whu_sar",
    "whu-opt-sar": "whu_sar",
    "yeseg": "yeseg_sar",
    "yeseg-opt-sar": "yeseg_sar",
    "pie": "pie_sar",
    "pie-rgb-sar": "pie_sar",
}

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG", ".TIF", ".TIFF")


def canonical_dataset_key(dataset: str) -> str:
    raw_key = dataset.strip().lower()
    key = DATASET_ALIASES.get(raw_key, raw_key).replace("-", "_")
    key = DATASET_ALIASES.get(key, key)
    if key not in SEG_SAR_DATASETS:
        valid = ", ".join(sorted(SEG_SAR_DATASETS))
        raise ValueError(f"Unknown SEG_SAR dataset '{dataset}'. Valid choices: {valid}")
    return key


def get_prompt_classes(cfg: SegSARDatasetConfig, yeseg_prompt_preset: str = "legacy") -> tuple[str, ...]:
    if cfg.key == "yeseg_sar":
        if yeseg_prompt_preset == "current":
            return YESEG_CURRENT_PROMPT_CLASSES
        if yeseg_prompt_preset == "legacy":
            return YESEG_LEGACY_PROMPT_CLASSES
        raise ValueError(f"Unknown YESeg prompt preset: {yeseg_prompt_preset}")
    prompt_classes = cfg.prompt_classes or cfg.classes
    if len(prompt_classes) != len(cfg.classes):
        raise ValueError(
            f"{cfg.display_name} has {len(cfg.classes)} label classes but "
            f"{len(prompt_classes)} prompt classes"
        )
    return prompt_classes


def get_eval_class_ids(cfg: SegSARDatasetConfig, include_background: bool) -> list[int]:
    if include_background:
        eval_class_ids = list(range(len(cfg.classes)))
    else:
        eval_class_ids = [
            idx
            for idx, class_name in enumerate(cfg.classes)
            if class_name.strip().lower() not in BACKGROUND_CLASS_NAMES
        ]
    if not eval_class_ids:
        raise ValueError(f"No evaluation classes left for {cfg.display_name}; check --include_background.")
    return eval_class_ids


def get_transform(img_size: int):
    transforms = [
        T.Resize(img_size, img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(transforms)


def class_aliases(class_name: str, use_alias_prompts: bool) -> list[str]:
    if not use_alias_prompts:
        return [class_name.replace(",", " ")]
    aliases = [alias.strip() for alias in class_name.split(",") if alias.strip()]
    return aliases or [class_name]


def build_text_prompts(classes: tuple[str, ...], tokenizer, max_tokens: int, use_alias_prompts: bool):
    prompt_ids: dict[str, list[torch.Tensor]] = {}
    prompt_attn: dict[str, list[torch.Tensor]] = {}
    prompt_text = build_prompt_text(classes, use_alias_prompts)
    for class_name in classes:
        prompt_ids[class_name] = []
        prompt_attn[class_name] = []
        for alias in class_aliases(class_name, use_alias_prompts):
            prompt = f"{alias.replace('_', ' ')} in the image."
            encoded = tokenizer(
                prompt,
                add_special_tokens=True,
                max_length=max_tokens,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            prompt_ids[class_name].append(encoded["input_ids"])
            prompt_attn[class_name].append(encoded["attention_mask"])
    return prompt_ids, prompt_attn, prompt_text


def build_prompt_text(classes: tuple[str, ...], use_alias_prompts: bool) -> dict[str, list[str]]:
    return {
        class_name: [f"{alias.replace('_', ' ')} in the image." for alias in class_aliases(class_name, use_alias_prompts)]
        for class_name in classes
    }


def forward_class(model, bert_model, image_tensor, input_ids, attention_mask, modal):
    if bert_model is not None:
        hidden_states = bert_model(input_ids, attention_mask=attention_mask)[0]
        embedding = hidden_states.permute(0, 2, 1).to(dtype=image_tensor.dtype)
        output = model(image_tensor, embedding, l_mask=attention_mask.unsqueeze(-1), modal=modal)
    else:
        output = model(image_tensor, input_ids, l_mask=attention_mask, modal=modal)
    return output


def foreground_probability(output: torch.Tensor) -> torch.Tensor:
    if output.shape[1] == 1:
        return torch.sigmoid(output)[:, 0]
    return torch.softmax(output, dim=1)[:, 1]


def resolve_path(base_dir: Path, entry: str, suffix: str, fallback_suffixes: tuple[str, ...] = IMAGE_SUFFIXES) -> Path:
    entry_path = Path(entry)
    candidates: list[Path] = []
    if entry_path.suffix:
        candidates.append(base_dir / entry_path)
    else:
        candidates.append(base_dir / f"{entry}{suffix}")
        candidates.extend(base_dir / f"{entry}{candidate_suffix}" for candidate_suffix in fallback_suffixes)

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No file found for '{entry}' under {base_dir}")


def strip_known_suffix(path: Path, suffix: str) -> str:
    suffix_lower = suffix.lower()
    path_str = path.as_posix()
    if path_str.lower().endswith(suffix_lower):
        return path_str[: -len(suffix)]
    return path.with_suffix("").as_posix()


def collect_samples(cfg: SegSARDatasetConfig, root: Path, limit: int | None = None) -> list[tuple[Path, Path, str]]:
    image_dir = root / cfg.image_dir
    label_dir = root / cfg.label_dir
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    entries: list[str]
    if cfg.ann_file is not None:
        ann_file = root / cfg.ann_file
        if not ann_file.is_file():
            raise FileNotFoundError(f"Annotation split file does not exist: {ann_file}")
        entries = [line.strip() for line in ann_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        label_files = sorted(path for path in label_dir.rglob(f"*{cfg.label_suffix}") if path.is_file())
        entries = [strip_known_suffix(path.relative_to(label_dir), cfg.label_suffix) for path in label_files]

    if limit is not None and limit > 0:
        entries = entries[:limit]

    samples = []
    missing = []
    for entry in entries:
        try:
            image_path = resolve_path(image_dir, entry, cfg.image_suffix)
            label_path = resolve_path(label_dir, entry, cfg.label_suffix)
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue
        samples.append((image_path, label_path, entry))

    if missing:
        print(f"Skipped {len(missing)} entries with missing files. Examples: {missing[:5]}")
    if not samples:
        raise FileNotFoundError(f"No matched image/label pairs found for dataset {cfg.display_name}")
    return samples


def load_label(label_path: Path, num_classes: int, reduce_zero_label: bool, ignore_class_ids: set[int]):
    label = Image.open(label_path)
    label_arr = np.array(label)
    if label_arr.ndim == 3:
        label_arr = label_arr[..., 0]
    label_arr = label_arr.astype(np.int64)

    if reduce_zero_label:
        label_arr = label_arr.copy()
        label_arr[label_arr == 0] = IGNORE_INDEX
        label_arr = label_arr - 1
        label_arr[label_arr == IGNORE_INDEX - 1] = IGNORE_INDEX

    valid = (label_arr >= 0) & (label_arr < num_classes)
    ignored = label_arr == IGNORE_INDEX
    unknown = ~(valid | ignored)

    mapped = np.full(label_arr.shape, IGNORE_INDEX, dtype=np.int64)
    mapped[valid] = label_arr[valid]
    ignored_class_pixels = 0
    if ignore_class_ids:
        ignored_class_mask = np.isin(mapped, list(ignore_class_ids))
        ignored_class_pixels = int(ignored_class_mask.sum())
        mapped[ignored_class_mask] = IGNORE_INDEX
    return mapped, int(unknown.sum()), int(ignored.sum()), ignored_class_pixels


def resize_pred_to_label(pred: np.ndarray, label_shape: tuple[int, int]) -> np.ndarray:
    if pred.shape == label_shape:
        return pred
    pred_image = Image.fromarray(pred.astype(np.uint8), mode="L")
    pred_image = TF.resize(pred_image, label_shape, interpolation=InterpolationMode.NEAREST)
    return np.array(pred_image, dtype=np.uint8)


def fast_hist(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    valid = (gt >= 0) & (gt < num_classes)
    return np.bincount(
        num_classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def parse_class_values(raw: str, num_classes: int, value_name: str) -> list[float] | None:
    if raw is None or raw.strip() == "":
        return None
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != num_classes:
        raise ValueError(f"--{value_name} expects {num_classes} comma-separated values, got {len(values)}")
    return values


def sanitize_result_tag(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw.strip())


def compute_metrics(hist: np.ndarray, mean_indices: list[int]) -> dict:
    intersection = np.diag(hist).astype(np.float64)
    gt_area = hist.sum(axis=1).astype(np.float64)
    pred_area = hist.sum(axis=0).astype(np.float64)
    union = gt_area + pred_area - intersection

    iou = safe_divide(intersection, union)
    acc = safe_divide(intersection, gt_area)
    all_acc = float(intersection.sum() / hist.sum() * 100.0) if hist.sum() > 0 else float("nan")
    return {
        "IoU": iou * 100.0,
        "Acc": acc * 100.0,
        "mIoU": float(np.nanmean(iou[mean_indices]) * 100.0),
        "mAcc": float(np.nanmean(acc[mean_indices]) * 100.0),
        "mIoU_all_non_nan": float(np.nanmean(iou) * 100.0),
        "mAcc_all_non_nan": float(np.nanmean(acc) * 100.0),
        "aAcc": all_acc,
        "intersection": intersection.astype(np.int64).tolist(),
        "union": union.astype(np.int64).tolist(),
        "gt_area": gt_area.astype(np.int64).tolist(),
        "pred_area": pred_area.astype(np.int64).tolist(),
    }


def predict_semantic_scores(
    model,
    bert_model,
    image_tensor: torch.Tensor,
    eval_class_ids: list[int],
    prompt_classes: tuple[str, ...],
    prompt_ids: dict[str, list[torch.Tensor]],
    prompt_attn: dict[str, list[torch.Tensor]],
    modal: torch.Tensor,
    img_size: int,
    device: torch.device,
    merge_strategy: str,
) -> torch.Tensor:
    class_maps = []
    for class_id in eval_class_ids:
        class_name = prompt_classes[class_id]
        alias_maps = []
        for ids, attn in zip(prompt_ids[class_name], prompt_attn[class_name]):
            output = forward_class(model, bert_model, image_tensor, ids.to(device), attn.to(device), modal)
            output = F.interpolate(output, (img_size, img_size), align_corners=True, mode="bilinear")
            if merge_strategy == "seg_hard_max":
                if output.shape[1] == 1:
                    alias_maps.append((torch.sigmoid(output)[:, 0] > 0.5).squeeze(0).to(torch.int64))
                else:
                    alias_maps.append((output.argmax(1) > 0).squeeze(0).to(torch.int64))
            else:
                alias_maps.append(foreground_probability(output).squeeze(0))
        class_maps.append(torch.stack(alias_maps, dim=0).max(dim=0).values)

    return torch.stack(class_maps, dim=0)


def semantic_scores_to_mask(
    scores: torch.Tensor,
    eval_class_ids: list[int],
    device: torch.device,
    merge_strategy: str,
    default_class_id: int,
    class_score_scale: list[float] | None = None,
    class_score_bias: list[float] | None = None,
    score_smooth_kernel: int = 0,
    mode_filter_size: int = 0,
) -> np.ndarray:
    if merge_strategy == "seg_hard_max":
        stacked = scores.bool()
        priority = torch.arange(len(eval_class_ids), device=device, dtype=torch.int64).view(-1, 1, 1)
        fallback = torch.full_like(priority, -1)
        winner = torch.where(stacked, priority, fallback).max(dim=0).values
        class_id_map = torch.tensor([default_class_id] + eval_class_ids, device=device, dtype=torch.int64)
        pred_tensor = class_id_map[(winner + 1).clamp(min=0)]
    else:
        scores = scores.to(device)
        if class_score_scale is not None:
            scale = torch.tensor(
                [class_score_scale[class_id] for class_id in eval_class_ids],
                device=device,
                dtype=scores.dtype,
            ).view(-1, 1, 1)
            scores = scores * scale
        if class_score_bias is not None:
            bias = torch.tensor(
                [class_score_bias[class_id] for class_id in eval_class_ids],
                device=device,
                dtype=scores.dtype,
            ).view(-1, 1, 1)
            scores = scores + bias
        if score_smooth_kernel and score_smooth_kernel > 1:
            if score_smooth_kernel % 2 == 0:
                raise ValueError("--score_smooth_kernel must be odd")
            scores = F.avg_pool2d(
                scores.unsqueeze(0),
                kernel_size=score_smooth_kernel,
                stride=1,
                padding=score_smooth_kernel // 2,
            ).squeeze(0)
        winner = scores.argmax(dim=0)
        class_id_map = torch.tensor(eval_class_ids, device=device, dtype=torch.int64)
        pred_tensor = class_id_map[winner]
    pred = pred_tensor.cpu().numpy().astype(np.uint8)
    if mode_filter_size and mode_filter_size > 1:
        if mode_filter_size % 2 == 0:
            raise ValueError("--mode_filter_size must be odd")
        pred = np.array(
            Image.fromarray(pred, mode="L").filter(ImageFilter.ModeFilter(size=mode_filter_size)),
            dtype=np.uint8,
        )
    return pred


def predict_semantic_mask(
    model,
    bert_model,
    image_tensor: torch.Tensor,
    eval_class_ids: list[int],
    prompt_classes: tuple[str, ...],
    prompt_ids: dict[str, list[torch.Tensor]],
    prompt_attn: dict[str, list[torch.Tensor]],
    modal: torch.Tensor,
    img_size: int,
    device: torch.device,
    merge_strategy: str,
    default_class_id: int,
    class_score_scale: list[float] | None = None,
    class_score_bias: list[float] | None = None,
    score_smooth_kernel: int = 0,
    mode_filter_size: int = 0,
) -> np.ndarray:
    scores = predict_semantic_scores(
        model,
        bert_model,
        image_tensor,
        eval_class_ids,
        prompt_classes,
        prompt_ids,
        prompt_attn,
        modal,
        img_size,
        device,
        merge_strategy,
    )
    return semantic_scores_to_mask(
        scores,
        eval_class_ids,
        device,
        merge_strategy,
        default_class_id,
        class_score_scale,
        class_score_bias,
        score_smooth_kernel,
        mode_filter_size,
    )


def evaluate(args):
    dataset_key = canonical_dataset_key(args.dataset)
    cfg = SEG_SAR_DATASETS[dataset_key]
    root = Path(args.seg_sar_root)
    img_size = cfg.eval_size if args.img_size <= 0 else args.img_size
    if getattr(args, "vision_backbone", "") == "dinov3_convnext_b" and img_size != 896:
        raise ValueError(
            "The current DINOv3 ConvNeXt RemoteSAM checkpoint uses a CIM block fixed to img_size=896. "
            f"Got img_size={img_size}. Use IMG_SIZE=896 for this checkpoint, or adapt the CIM block "
            "before using the official SegEarth-OV-2 resize size."
        )
    args.img_size = img_size

    cudnn.benchmark = True
    cudnn.deterministic = True

    samples = collect_samples(cfg, root, limit=args.limit_samples)
    prompt_classes = get_prompt_classes(cfg, yeseg_prompt_preset=args.yeseg_prompt_preset)
    eval_class_ids = get_eval_class_ids(cfg, include_background=args.include_background)
    ignored_class_ids = set(range(len(cfg.classes))) - set(eval_class_ids)
    default_class_id = next(iter(sorted(ignored_class_ids)), eval_class_ids[0])

    score_cache = None
    cached_scores = None
    cached_entries: list[str] = []
    using_score_cache = bool(args.load_score_cache)
    if using_score_cache:
        score_cache = np.load(args.load_score_cache, allow_pickle=True)
        cached_scores = score_cache["scores"]
        cached_entries = [str(item) for item in score_cache["entries"].tolist()]
        if len(cached_scores) < len(samples):
            raise ValueError(
                f"Score cache has {len(cached_scores)} samples but evaluation needs {len(samples)} samples"
            )
        device = torch.device("cpu")
        model = None
        bert_model = None
        prompt_ids = {}
        prompt_attn = {}
        prompt_text = build_prompt_text(prompt_classes, use_alias_prompts=not args.no_alias_prompts)
        transform = None
    else:
        model, bert_model, device = load_model_and_text_encoder(args, distributed=False)
        model.eval()
        if bert_model is not None:
            bert_model.eval()

        tokenizer_path = (
            args.modernbert_path if getattr(args, "text_encoder", "modernbert") == "modernbert" else args.bert_tokenizer
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        prompt_ids, prompt_attn, prompt_text = build_text_prompts(
            prompt_classes,
            tokenizer,
            max_tokens=args.max_tokens,
            use_alias_prompts=not args.no_alias_prompts,
        )
        transform = get_transform(img_size)

    num_classes = len(cfg.classes)
    class_score_scale = parse_class_values(args.class_score_scale, num_classes, "class_score_scale")
    class_score_bias = parse_class_values(args.class_score_bias, num_classes, "class_score_bias")
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_pixels = 0
    ignored_pixels = 0
    ignored_class_pixels = 0
    unknown_pixels = 0
    modal = torch.tensor([1 if getattr(args, "modal", "sar") == "sar" else 0], device=device, dtype=torch.long)
    per_sample_results = []
    score_cache_scores = []
    score_cache_entries = []
    score_cache_label_paths = []

    print(f"Evaluating {cfg.display_name} ({dataset_key})")
    print(f"Dataset root: {root}")
    print(f"Samples: {len(samples)}")
    print(f"Image size: {img_size}")
    print(f"Classes ({num_classes}): " + ", ".join(f"{idx}:{name}" for idx, name in enumerate(cfg.classes)))
    print("Prompt classes: " + ", ".join(f"{idx}:{name}" for idx, name in enumerate(prompt_classes)))
    print("Eval classes: " + ", ".join(f"{idx}:{cfg.classes[idx]}" for idx in eval_class_ids))
    if ignored_class_ids:
        print("Ignored configured classes: " + ", ".join(f"{idx}:{cfg.classes[idx]}" for idx in sorted(ignored_class_ids)))
    print(f"reduce_zero_label: {cfg.reduce_zero_label}")
    print(f"Merge strategy: {args.merge_strategy}")
    if args.load_score_cache:
        print(f"Loading score cache: {args.load_score_cache}")
    if args.save_score_cache:
        print(f"Saving score cache: {args.save_score_cache}")
    if class_score_scale is not None:
        print("Class score scale: " + ", ".join(f"{idx}:{value:g}" for idx, value in enumerate(class_score_scale)))
    if class_score_bias is not None:
        print("Class score bias: " + ", ".join(f"{idx}:{value:g}" for idx, value in enumerate(class_score_bias)))
    if args.score_smooth_kernel > 1:
        print(f"Score smoothing kernel: {args.score_smooth_kernel}")
    if args.mode_filter_size > 1:
        print(f"Mode filter size: {args.mode_filter_size}")
    print(f"Metric: mIoU over {len(eval_class_ids)} eval classes, ignore_index={IGNORE_INDEX}")

    with torch.no_grad():
        for sample_index, (image_path, label_path, entry) in enumerate(tqdm(samples, desc=f"Evaluating {cfg.display_name}")):
            gt, unknown, ignored, ignored_configured = load_label(
                label_path,
                num_classes,
                cfg.reduce_zero_label,
                ignored_class_ids,
            )
            unknown_pixels += unknown
            ignored_pixels += ignored
            ignored_class_pixels += ignored_configured
            total_pixels += int(gt.size)

            if using_score_cache:
                if cached_entries and cached_entries[sample_index] != entry:
                    raise ValueError(
                        f"Score cache entry mismatch at {sample_index}: "
                        f"cache={cached_entries[sample_index]!r}, dataset={entry!r}"
                    )
                scores = torch.from_numpy(cached_scores[sample_index].astype(np.float32))
            else:
                image = Image.open(image_path).convert("RGB")
                image_tensor, _ = transform(image, image)
                image_tensor = image_tensor.unsqueeze(0).to(device)
                scores = predict_semantic_scores(
                    model,
                    bert_model,
                    image_tensor,
                    eval_class_ids,
                    prompt_classes,
                    prompt_ids,
                    prompt_attn,
                    modal,
                    img_size,
                    device,
                    args.merge_strategy,
                )
                if args.save_score_cache:
                    cached = F.interpolate(
                        scores.float().unsqueeze(0),
                        size=gt.shape,
                        align_corners=True,
                        mode="bilinear",
                    ).squeeze(0)
                    score_cache_scores.append(cached.cpu().numpy().astype(np.float16))
                    score_cache_entries.append(entry)
                    score_cache_label_paths.append(str(label_path))

            pred = semantic_scores_to_mask(
                scores,
                eval_class_ids,
                device,
                args.merge_strategy,
                default_class_id,
                class_score_scale,
                class_score_bias,
                args.score_smooth_kernel,
                args.mode_filter_size,
            )
            pred = resize_pred_to_label(pred, gt.shape)
            sample_hist = fast_hist(gt, pred, num_classes)
            hist += sample_hist

            if args.save_per_sample:
                sample_metrics = compute_metrics(sample_hist, eval_class_ids)
                per_sample_results.append(
                    {
                        "entry": entry,
                        "image": str(image_path),
                        "label": str(label_path),
                        "mIoU": sample_metrics["mIoU"],
                        "mPA": sample_metrics["mAcc"],
                        "aAcc": sample_metrics["aAcc"],
                        "gt_area": sample_metrics["gt_area"],
                        "pred_area": sample_metrics["pred_area"],
                        "intersection": sample_metrics["intersection"],
                        "union": sample_metrics["union"],
                        "per_class_iou": {
                            cfg.classes[idx]: None
                            if np.isnan(sample_metrics["IoU"][idx])
                            else float(sample_metrics["IoU"][idx])
                            for idx in range(num_classes)
                        },
                        "hist": sample_hist.tolist(),
                    }
                )

    metrics = compute_metrics(hist, eval_class_ids)

    print(f"{cfg.display_name} semantic segmentation results:")
    for idx, class_name in enumerate(cfg.classes):
        iou = metrics["IoU"][idx]
        acc = metrics["Acc"][idx]
        iou_text = "nan" if np.isnan(iou) else f"{iou:.2f}"
        acc_text = "nan" if np.isnan(acc) else f"{acc:.2f}"
        suffix = "" if idx in eval_class_ids else " (excluded from mean)"
        print(f"===>{idx}:{class_name}\tmIoU-{iou_text}; mPA-{acc_text}{suffix}")
    print(f"===> mIoU: {metrics['mIoU']:.2f}; mPA: {metrics['mAcc']:.2f}; aAcc: {metrics['aAcc']:.2f}")

    if ignored_pixels:
        print(f"Ignored pixels: {ignored_pixels} ({ignored_pixels * 100.0 / max(total_pixels, 1):.4f}%)")
    if ignored_class_pixels:
        print(
            f"Ignored configured-class pixels: {ignored_class_pixels} "
            f"({ignored_class_pixels * 100.0 / max(total_pixels, 1):.4f}%)"
        )
    if unknown_pixels:
        print(f"Ignored unknown-label pixels: {unknown_pixels} ({unknown_pixels * 100.0 / max(total_pixels, 1):.4f}%)")

    if args.save_score_cache:
        cache_path = Path(args.save_score_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            scores=np.stack(score_cache_scores, axis=0),
            entries=np.array(score_cache_entries),
            label_paths=np.array(score_cache_label_paths),
            eval_class_ids=np.array(eval_class_ids, dtype=np.int64),
            classes=np.array(cfg.classes),
            prompt_classes=np.array(prompt_classes),
            dataset=np.array(dataset_key),
            img_size=np.array(img_size, dtype=np.int64),
            merge_strategy=np.array(args.merge_strategy),
        )
        print(f"Score cache saved to {cache_path}")

    result = {
        "dataset": dataset_key,
        "display_name": cfg.display_name,
        "mIoU": metrics["mIoU"],
        "mPA": metrics["mAcc"],
        "mIoU_all_non_nan": metrics["mIoU_all_non_nan"],
        "mPA_all_non_nan": metrics["mAcc_all_non_nan"],
        "aAcc": metrics["aAcc"],
        "metric_note": f"mIoU over configured eval classes {eval_class_ids}; ignore_index={IGNORE_INDEX}",
        "prediction_note": "Predictions are resized back to each label's original shape before accumulating the confusion matrix.",
        "classes": {
            class_name: {
                "id": idx,
                "used_for_mean": idx in eval_class_ids,
                "IoU": None if np.isnan(metrics["IoU"][idx]) else float(metrics["IoU"][idx]),
                "PA": None if np.isnan(metrics["Acc"][idx]) else float(metrics["Acc"][idx]),
                "prompt_class": prompt_classes[idx],
                "prompts": prompt_text[prompt_classes[idx]],
            }
            for idx, class_name in enumerate(cfg.classes)
        },
        "hist": hist.tolist(),
        "areas": {
            "intersection": metrics["intersection"],
            "union": metrics["union"],
            "gt_area": metrics["gt_area"],
            "pred_area": metrics["pred_area"],
        },
        "ignored_pixels": ignored_pixels,
        "ignored_class_pixels": ignored_class_pixels,
        "unknown_pixels": unknown_pixels,
        "total_pixels": total_pixels,
        "num_samples": len(samples),
        "img_size": img_size,
        "modal": getattr(args, "modal", "sar"),
        "include_background": args.include_background,
        "eval_class_ids": eval_class_ids,
        "ignored_class_ids": sorted(ignored_class_ids),
        "merge_strategy": args.merge_strategy,
        "load_score_cache": args.load_score_cache or None,
        "save_score_cache": args.save_score_cache or None,
        "yeseg_prompt_preset": args.yeseg_prompt_preset if dataset_key == "yeseg_sar" else None,
        "postprocess": {
            "class_score_scale": class_score_scale,
            "class_score_bias": class_score_bias,
            "score_smooth_kernel": args.score_smooth_kernel,
            "mode_filter_size": args.mode_filter_size,
        },
        "config": asdict(cfg),
        "root": str(root),
    }

    if args.save_path:
        save_dir = Path(args.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        result_tag = sanitize_result_tag(args.result_tag)
        if not result_tag and args.limit_samples is not None:
            result_tag = f"limit{args.limit_samples}"
        output_stem = f"seg_sar_{dataset_key}" + (f"_{result_tag}" if result_tag else "")
        output_path = save_dir / f"{output_stem}_results.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Results saved to {output_path}")
        if args.save_per_sample:
            per_sample_path = save_dir / f"{output_stem}_per_sample.json"
            per_sample_path.write_text(json.dumps(per_sample_results, indent=2), encoding="utf-8")
            print(f"Per-sample diagnostics saved to {per_sample_path}")

    return result


def main(args):
    evaluate(args)


if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    parser.add_argument(
        "--seg_sar_root",
        default=DEFAULT_SEG_SAR_ROOT,
        help="Root directory containing the SegEarth-OV-2 SAR datasets.",
    )
    parser.add_argument(
        "--limit_samples",
        default=0,
        type=int,
        help="Evaluate only the first N matched samples; 0 means all samples.",
    )
    parser.add_argument(
        "--no_alias_prompts",
        action="store_true",
        help="Use one prompt per official class string instead of splitting comma-separated aliases.",
    )
    parser.add_argument(
        "--include_background",
        action="store_true",
        help="Include classes named 'background' in prompts and mIoU. By default, background classes are ignored.",
    )
    parser.add_argument(
        "--merge_strategy",
        default="prob_argmax",
        choices=("prob_argmax", "seg_hard_max"),
        help=(
            "How to merge per-class binary segmentation into a semantic mask. "
            "prob_argmax compares foreground probabilities; seg_hard_max mirrors SEG.py's hard-mask max merge."
        ),
    )
    parser.add_argument(
        "--yeseg_prompt_preset",
        default="legacy",
        choices=("legacy", "current"),
        help="Prompt wording preset for YESeg-OPT-SAR. legacy restores the earlier alias prompts.",
    )
    parser.add_argument(
        "--class_score_scale",
        default="",
        help="Comma-separated per-class multiplicative factors applied before prob_argmax, e.g. '1,1,1,1,1,1'.",
    )
    parser.add_argument(
        "--class_score_bias",
        default="",
        help="Comma-separated per-class additive biases applied before prob_argmax, e.g. '0,0,0,0,0,0'.",
    )
    parser.add_argument(
        "--score_smooth_kernel",
        default=0,
        type=int,
        help="Odd kernel size for average smoothing of per-class score maps before argmax; 0 disables it.",
    )
    parser.add_argument(
        "--mode_filter_size",
        default=0,
        type=int,
        help="Odd kernel size for PIL mode filtering after argmax; 0 disables it.",
    )
    parser.add_argument(
        "--save_per_sample",
        action="store_true",
        help="Save per-image confusion matrices and metrics for debugging dataset-specific failures.",
    )
    parser.add_argument(
        "--save_score_cache",
        default="",
        help="Optional .npz path for saving per-class foreground score maps resized to label resolution.",
    )
    parser.add_argument(
        "--load_score_cache",
        default="",
        help="Optional .npz path for evaluating postprocess settings without re-running the model.",
    )
    parser.add_argument(
        "--result_tag",
        default="",
        help="Optional suffix added to result filenames, useful when comparing postprocess settings.",
    )
    args = parser.parse_args()
    if args.limit_samples <= 0:
        args.limit_samples = None
    main(args)
