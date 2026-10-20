#!/usr/bin/env python3
"""Small CPU checks of recovered native gate and metrics, not paper reproduction."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from lib.modal_inference import infer_modal_id_from_image
    from tasks.code._adapters.MLC_ranking import multilabel_ranking_metrics
    from tasks.code.model_auto import RemoteSAM

    assert infer_modal_id_from_image(np.zeros((8, 8), dtype=np.uint8)) == 1
    assert infer_modal_id_from_image(np.full((8, 8, 3), 128, dtype=np.uint8)) == 1
    color = np.zeros((8, 8, 3), dtype=np.uint8)
    color[..., 0] = 255
    assert infer_modal_id_from_image(color) == 0
    metrics = multilabel_ranking_metrics([
        {"image": "a", "gt_classes": ["ship"], "scores": {"ship": .9, "bridge": .1}},
        {"image": "b", "gt_classes": ["bridge"], "scores": {"ship": .1, "bridge": .9}},
    ], ["ship", "bridge"])
    assert metrics["mAP"] == 1.0 and metrics["micro_f1"] == 1.0
    operations = ("referring_seg", "semantic_seg", "visual_grounding", "detection", "multi_label_cls", "multi_class_cls", "captioning", "counting")
    assert all(callable(getattr(RemoteSAM, name, None)) for name in operations)
    report = {"gate_cases_passed": 3, "native_ranking_metric_sanity_passed": True,
              "recovered_task_operations": list(operations),
              "scope": "Small CPU sanity fixtures; not full dataset scores or numerical parity proof."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
