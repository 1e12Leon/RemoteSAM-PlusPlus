#!/usr/bin/env python3
"""Use the existing eight task transitions with an explicitly selected modality."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint", required=True)
    cli.add_argument("--image", required=True)
    cli.add_argument("--modal", required=True, choices=("rgb", "sar"))
    cli.add_argument("--task", required=True, choices=("REF", "SEG", "VG", "DET", "MLC", "MCC", "CAP", "CNT"))
    cli.add_argument("--sentence", default="")
    cli.add_argument("--classes", nargs="+", default=[])
    cli.add_argument("--device", default="cuda:0")
    cli.add_argument("--epoc", action="store_true")
    cli.add_argument("--output-dir", required=True)
    cli.add_argument("--modernbert-path", default=str(ROOT / "gte-modernbert-base"))
    cli.add_argument("--dinov3-path", default=str(ROOT / "dinov3-convnext-B"))
    options = cli.parse_args()
    if options.task in ("REF", "VG") and not options.sentence:
        cli.error("--sentence is required for REF/VG")
    if options.task not in ("REF", "VG") and not options.classes:
        cli.error("--classes is required for this task")

    import numpy as np
    from PIL import Image
    import torch
    from args import get_parser
    from tasks.code.model import RemoteSAM
    from tasks.code.runtime import load_model_and_text_encoder

    config = json.loads((ROOT / "configs/paper_model.json").read_text(encoding="utf-8"))
    args = get_parser().parse_args([])
    for name, value in config.items():
        setattr(args, name, value)
    args.resume = options.checkpoint
    args.modal = options.modal
    args.device = options.device
    args.modernbert_path = options.modernbert_path
    args.dinov3_convnext_path = options.dinov3_path
    model, text_encoder, device = load_model_and_text_encoder(args)
    predictor = RemoteSAM(model, device, args=args, modal=options.modal,
                          text_encoder=text_encoder, use_EPOC=options.epoc)
    image = np.asarray(Image.open(options.image).convert("RGB"))
    functions = {
        "REF": lambda: predictor.referring_seg(image=image, sentence=options.sentence),
        "SEG": lambda: predictor.semantic_seg(image=image, classnames=options.classes),
        "VG": lambda: predictor.visual_grounding(image=image, sentence=options.sentence),
        "DET": lambda: predictor.detection(image=image, classnames=options.classes),
        "MLC": lambda: predictor.multi_label_cls(image=image, classnames=options.classes, return_prob=True),
        "MCC": lambda: predictor.multi_class_cls(image=image, classnames=options.classes, return_prob=True),
        "CAP": lambda: predictor.captioning(image=image, classnames=options.classes),
        "CNT": lambda: predictor.counting(image=image, classnames=options.classes),
    }
    with torch.no_grad():
        result = functions[options.task]()
    destination = Path(options.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if options.task == "REF":
        np.save(destination / "mask.npy", result)
        result = {"mask": "mask.npy", "shape": list(result.shape)}
    elif options.task == "SEG":
        masks = {}
        for index, (name, mask) in enumerate(result.items()):
            filename = f"mask_{index:03d}.npy"
            np.save(destination / filename, mask)
            masks[name] = filename
        result = masks

    def json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported result type: {type(value).__name__}")

    manifest = {"task": options.task, "modal": options.modal, "image": options.image, "result": result}
    (destination / "result.json").write_text(json.dumps(manifest, default=json_default, indent=2), encoding="utf-8")
    print(f"Saved {options.task}/{options.modal} output to {destination.resolve()}")


if __name__ == "__main__":
    main()
