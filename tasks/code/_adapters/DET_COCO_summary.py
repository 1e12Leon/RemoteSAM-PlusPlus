import argparse
import contextlib
import io
import json
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def category_ids_for_classes(coco_gt, classes):
    name_to_id = {item["name"]: item["id"] for item in coco_gt.dataset.get("categories", [])}
    missing = [class_name for class_name in classes if class_name not in name_to_id]
    if missing:
        raise ValueError(f"Classes not found in annotation categories: {' '.join(missing)}")
    return [name_to_id[class_name] for class_name in classes]


def load_predictions(pred_dir, classes):
    merged = []
    for class_name in classes:
        pred_path = os.path.join(pred_dir, f"preds_{class_name}.json")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing prediction file: {pred_path}")
        with open(pred_path, "r") as handle:
            predictions = json.load(handle)
        if not isinstance(predictions, list):
            raise ValueError(f"Prediction file must contain a JSON list: {pred_path}")
        merged.extend(predictions)
    merged.sort(key=lambda item: (item["image_id"], item["category_id"], -item["score"]))
    return merged


def summarize_metrics(coco_eval):
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()
    return {
        "mAP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
    }


def evaluate(anno_path, pred_path, classes):
    coco_gt = COCO(anno_path)
    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.catIds = category_ids_for_classes(coco_gt, classes)
    coco_eval.evaluate()
    coco_eval.accumulate()
    return coco_gt, coco_dt, summarize_metrics(coco_eval)


def evaluate_per_class(coco_gt, coco_dt, classes):
    class_ids = category_ids_for_classes(coco_gt, classes)
    rows = []
    for class_name, class_id in zip(classes, class_ids):
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.catIds = [class_id]
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.evaluate()
            coco_eval.accumulate()
        metrics = summarize_metrics(coco_eval)
        rows.append((class_name, metrics))
    return rows


def main(args):
    os.makedirs(args.pred_dir, exist_ok=True)
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.pred_dir, output_path)

    predictions = load_predictions(args.pred_dir, args.classes)
    with open(output_path, "w") as handle:
        json.dump(predictions, handle)

    print("\n########## COCO Detection Summary ##########")
    print(f"Classes: {' '.join(args.classes)}")
    print(f"Merged detections: {len(predictions)}")
    print(f"Prediction file: {output_path}")

    if len(predictions) == 0:
        print("No detections found. COCOeval cannot compute meaningful AP.")
        return

    coco_gt, coco_dt, metrics = evaluate(args.annoFolder, output_path, args.classes)
    print("Comparable total metrics:")
    print("mAP  = {:.3f}".format(metrics["mAP"]))
    print("AP50 = {:.3f}".format(metrics["AP50"]))
    print("AP75 = {:.3f}".format(metrics["AP75"]))
    print("mAP/AP50/AP75 (%) = {:.1f}/{:.1f}/{:.1f}".format(
        metrics["mAP"] * 100.0,
        metrics["AP50"] * 100.0,
        metrics["AP75"] * 100.0,
    ))
    print("Per-class metrics (%):")
    for class_name, row_metrics in evaluate_per_class(coco_gt, coco_dt, args.classes):
        print("{:<12s} mAP={:>5.1f} AP50={:>5.1f} AP75={:>5.1f}".format(
            class_name,
            row_metrics["mAP"] * 100.0,
            row_metrics["AP50"] * 100.0,
            row_metrics["AP75"] * 100.0,
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge DET per-class predictions and report total COCO bbox mAP/AP50/AP75."
    )
    parser.add_argument("--annoFolder", required=True, help="COCO-format annotation JSON.")
    parser.add_argument("--pred_dir", required=True, help="Directory containing preds_<class>.json files.")
    parser.add_argument("--classes", nargs="+", required=True, help="Class names to merge in evaluation order.")
    parser.add_argument("--output", default="preds_all.json", help="Merged COCO result JSON path or filename.")
    main(parser.parse_args())
