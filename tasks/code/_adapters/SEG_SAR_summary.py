from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_DATASETS = (
    "openearthmap_sar",
    "ddhrnet_korea_sar",
    "ddhrnet_shandong_sar",
    "ddhrnet_xian_sar",
    "fusar_sar",
    "whu_sar",
    "yeseg_sar",
    "pie_sar",
)


def format_metric(value):
    if value is None:
        return "nan"
    return f"{float(value):.2f}"


def load_result(result_dir: Path, dataset: str) -> dict:
    path = result_dir / f"seg_sar_{dataset}_results.json"
    if not path.is_file():
        return {
            "dataset": dataset,
            "status": "missing",
            "path": str(path),
        }

    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "dataset": dataset,
        "display_name": data.get("display_name", dataset),
        "status": "ok",
        "mIoU": data.get("mIoU"),
        "mPA": data.get("mPA"),
        "aAcc": data.get("aAcc"),
        "num_samples": data.get("num_samples"),
        "img_size": data.get("img_size"),
        "include_background": data.get("include_background"),
        "merge_strategy": data.get("merge_strategy"),
        "eval_class_ids": data.get("eval_class_ids"),
        "path": str(path),
    }


def print_table(rows: list[dict]) -> None:
    headers = ("dataset", "mIoU", "mPA", "aAcc", "samples", "merge", "bg")
    print("\t".join(headers))
    for row in rows:
        if row["status"] != "ok":
            print(f"{row['dataset']}\tmissing\tmissing\tmissing\tmissing\tmissing\tmissing")
            continue
        print(
            "\t".join(
                [
                    row["dataset"],
                    format_metric(row.get("mIoU")),
                    format_metric(row.get("mPA")),
                    format_metric(row.get("aAcc")),
                    str(row.get("num_samples")),
                    str(row.get("merge_strategy")),
                    str(row.get("include_background")),
                ]
            )
        )


def save_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "dataset",
        "display_name",
        "status",
        "mIoU",
        "mPA",
        "aAcc",
        "num_samples",
        "img_size",
        "include_background",
        "merge_strategy",
        "eval_class_ids",
        "path",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SEG_SAR result JSON files.")
    parser.add_argument("--result_dir", default="./result/SEG_SAR", help="Directory containing SEG_SAR result JSON files.")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS, help="Datasets to summarize in order.")
    parser.add_argument("--csv_path", default="./result/SEG_SAR/SEG_SAR_summary.csv", help="Where to save the CSV summary.")
    parser.add_argument("--json_path", default="./result/SEG_SAR/SEG_SAR_summary.json", help="Where to save the JSON summary.")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    rows = [load_result(result_dir, dataset) for dataset in args.datasets]

    print_table(rows)

    csv_path = Path(args.csv_path)
    json_path = Path(args.json_path)
    save_csv(rows, csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Summary CSV saved to {csv_path}")
    print(f"Summary JSON saved to {json_path}")


if __name__ == "__main__":
    main()
