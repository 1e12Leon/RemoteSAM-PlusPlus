import argparse
import contextlib
import io
import json
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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


def evaluate(anno_path, pred_path):
    coco_gt = COCO(anno_path)
    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()
    return {
        "mAP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
    }


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

    metrics = evaluate(args.annoFolder, output_path)
    print("Comparable total metrics:")
    print("mAP  = {:.3f}".format(metrics["mAP"]))
    print("AP50 = {:.3f}".format(metrics["AP50"]))
    print("AP75 = {:.3f}".format(metrics["AP75"]))
    print("mAP/AP50/AP75 (%) = {:.1f}/{:.1f}/{:.1f}".format(
        metrics["mAP"] * 100.0,
        metrics["AP50"] * 100.0,
        metrics["AP75"] * 100.0,
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
