from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert DIOR COCO annotations to CNT task JSON.")
    parser.add_argument("--coco-json", required=True, help="Path to DIOR COCO annotation JSON.")
    parser.add_argument("--output-json", required=True, help="Output path for CNT annotation JSON.")
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="Emit one sample for every image/category pair, including zero-count classes.",
    )
    return parser.parse_args()


def normalize_dior_class_name(name: str) -> str:
    return name.strip().lower()


def main():
    args = parse_args()
    coco_path = Path(args.coco_json)
    output_path = Path(args.output_json)

    with coco_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images = sorted(coco["images"], key=lambda item: item["id"])
    categories = sorted(coco["categories"], key=lambda item: item["id"])
    image_id_to_name = {item["id"]: item["file_name"] for item in images}
    category_id_to_name = {item["id"]: normalize_dior_class_name(item["name"]) for item in categories}

    counts = defaultdict(int)
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        if image_id not in image_id_to_name:
            raise KeyError(f"Annotation references unknown image_id={image_id}")
        if category_id not in category_id_to_name:
            raise KeyError(f"Annotation references unknown category_id={category_id}")
        counts[(image_id, category_id)] += 1

    records = []
    for image in images:
        image_id = image["id"]
        file_name = image["file_name"]
        for category in categories:
            category_id = category["id"]
            category_name = normalize_dior_class_name(category["name"])
            count = counts[(image_id, category_id)]
            if count == 0 and not args.include_zero:
                continue
            records.append(
                {
                    "images": [file_name],
                    "conversations": [
                        {"content": f"What is the number of {category_name}."},
                        {"content": str(count)},
                    ],
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False)

    total_count = sum(counts.values())
    emitted_count = sum(int(item["conversations"][1]["content"]) for item in records)
    print(f"images={len(images)}")
    print(f"categories={len(categories)}")
    print(f"annotations={len(coco['annotations'])}")
    print(f"records={len(records)}")
    print(f"sum_counts={emitted_count}")
    print(f"original_annotation_count={total_count}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
