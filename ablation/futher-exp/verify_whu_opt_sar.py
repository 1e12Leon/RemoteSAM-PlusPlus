#!/usr/bin/env python3
"""Verify that the downloaded WHU-OPT-SAR data matches the prepared test set.

This is intentionally a read-only protocol check. It reproduces the crop order
used by the existing SegEarth-OV-2 preparation and compares SAR pixels and the
0/10/.../70 -> 0/1/.../7 label conversion exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_RAW_LABELS = set(range(0, 71, 10))


def crop_starts(length: int, crop_size: int) -> list[int]:
    if length <= crop_size:
        return [0]
    starts = list(range(0, length - crop_size + 1, crop_size))
    final_start = length - crop_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def read_split(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [name for name in names if name]


def verify(args: argparse.Namespace) -> None:
    raw_sar_dir = args.raw_root / "sar"
    raw_label_dir = args.raw_root / "lbl"
    prepared_sar_dir = args.prepared_root / "sar_dir"
    prepared_label_dir = args.prepared_root / "ann_dir"

    required_dirs = (raw_sar_dir, raw_label_dir, prepared_sar_dir, prepared_label_dir)
    for directory in required_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"Required directory not found: {directory}")
    if not args.split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {args.split_file}")

    raw_sar_names = {path.name for path in raw_sar_dir.glob("*.tif")}
    raw_label_names = {path.name for path in raw_label_dir.glob("*.tif")}
    if raw_sar_names != raw_label_names:
        missing_labels = sorted(raw_sar_names - raw_label_names)[:10]
        missing_images = sorted(raw_label_names - raw_sar_names)[:10]
        raise ValueError(
            "Raw SAR/label filenames differ; "
            f"missing labels={missing_labels}, missing SAR images={missing_images}"
        )

    split_names = read_split(args.split_file)
    if args.max_images > 0:
        split_names = split_names[: args.max_images]

    checked_crops = 0
    for name in split_names:
        if name not in raw_sar_names:
            raise FileNotFoundError(f"Split entry is absent from raw data: {name}")

        sar = np.asarray(Image.open(raw_sar_dir / name))
        raw_label = np.asarray(Image.open(raw_label_dir / name))
        if sar.ndim != 2 or raw_label.ndim != 2:
            raise ValueError(f"Expected single-channel TIFFs for {name}")
        if sar.shape != raw_label.shape:
            raise ValueError(f"SAR/label size mismatch for {name}: {sar.shape} vs {raw_label.shape}")

        label_values = set(np.unique(raw_label).tolist())
        if not label_values.issubset(EXPECTED_RAW_LABELS):
            raise ValueError(f"Unexpected raw labels in {name}: {sorted(label_values)}")

        stem = Path(name).stem
        index = 0
        # The official prepared files advance vertically first, then horizontally.
        for x in crop_starts(sar.shape[1], args.crop_size):
            for y in crop_starts(sar.shape[0], args.crop_size):
                crop_name = f"{stem}_{index:03d}.tif"
                expected_sar = sar[y : y + args.crop_size, x : x + args.crop_size]
                expected_label = raw_label[y : y + args.crop_size, x : x + args.crop_size] // 10

                actual_sar_path = prepared_sar_dir / crop_name
                actual_label_path = prepared_label_dir / crop_name
                if not actual_sar_path.is_file() or not actual_label_path.is_file():
                    raise FileNotFoundError(f"Prepared pair missing: {crop_name}")

                actual_sar = np.asarray(Image.open(actual_sar_path))
                actual_label = np.asarray(Image.open(actual_label_path))
                if not np.array_equal(actual_sar, expected_sar):
                    raise ValueError(f"SAR crop mismatch: {actual_sar_path}")
                if not np.array_equal(actual_label, expected_label):
                    raise ValueError(f"Label crop mismatch: {actual_label_path}")

                checked_crops += 1
                index += 1

    print(f"Raw pairs: {len(raw_sar_names)}")
    print(f"Verified split images: {len(split_names)}")
    print(f"Verified prepared crop pairs: {checked_crops}")
    print("WHU-OPT-SAR raw data and SegEarth-OV-2 prepared test set match exactly.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/data/user2/SAR_Data/WHU-OPT-SAR"),
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("/data/user2/SAR_Data/SegEarthOV2/whu-opt-sar/crop512/test"),
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("/data/user2/SAR_Data/SegEarthOV2/whu-opt-sar/splits/test_list.txt"),
    )
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Verify only the first N source images; 0 verifies the full test split.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    verify(parse_args())

