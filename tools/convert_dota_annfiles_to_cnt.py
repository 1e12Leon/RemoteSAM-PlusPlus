from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DOTA_TO_CNT = {
    "plane": "airplane",
    "storage-tank": "storagetank",
    "baseball-diamond": "baseballfield",
    "tennis-court": "tenniscourt",
    "basketball-court": "basketballcourt",
    "ground-track-field": "groundtrackfield",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert DOTA txt annfiles to CNT task JSON.")
    parser.add_argument("--image-dir", required=True, help="Folder containing DOTA patch images.")
    parser.add_argument("--ann-dir", required=True, help="Folder containing DOTA patch txt annfiles.")
    parser.add_argument("--output-json", required=True, help="Output path for CNT annotation JSON.")
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="Emit one sample for every image/category pair, including zero-count classes.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=[
            "airplane",
            "ship",
            "storagetank",
            "baseballfield",
            "tenniscourt",
            "basketballcourt",
            "groundtrackfield",
            "harbor",
            "bridge",
            "large-vehicle",
            "small-vehicle",
            "helicopter",
            "roundabout",
            "soccer-ball-field",
            "swimming-pool",
            "container-crane",
            "airport",
            "helipad",
        ],
        help="CNT classes to emit after DOTA name normalization.",
    )
    return parser.parse_args()


def normalize_class_name(name: str) -> str:
    name = name.strip().lower()
    return DOTA_TO_CNT.get(name, name)


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    ann_dir = Path(args.ann_dir)
    output_path = Path(args.output_json)
    classes = list(args.classes)

    images = sorted(
        p.name for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found under: {image_dir}")
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation folder not found: {ann_dir}")

    records = []
    total_objects = 0
    unknown = Counter()
    class_totals = Counter()
    positive_pairs = 0

    for image_name in images:
        ann_path = ann_dir / f"{Path(image_name).stem}.txt"
        counts = Counter()
        if ann_path.is_file():
            with ann_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    classname = normalize_class_name(parts[8])
                    if classname not in classes:
                        unknown[parts[8]] += 1
                        continue
                    counts[classname] += 1
                    class_totals[classname] += 1
                    total_objects += 1

        for classname in classes:
            count = counts[classname]
            if count == 0 and not args.include_zero:
                continue
            positive_pairs += int(count > 0)
            records.append(
                {
                    "images": [image_name],
                    "conversations": [
                        {"content": f"What is the number of {classname}."},
                        {"content": str(count)},
                    ],
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False)

    print(f"images={len(images)}")
    print(f"classes={len(classes)}")
    print(f"records={len(records)}")
    print(f"positive_image_class_pairs={positive_pairs}")
    print(f"sum_counts={sum(int(item['conversations'][1]['content']) for item in records)}")
    print(f"matched_objects={total_objects}")
    if unknown:
        print(f"unknown_classes={dict(sorted(unknown.items()))}")
    print(f"class_totals={dict(sorted(class_totals.items()))}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
