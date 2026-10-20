#!/usr/bin/env python3
"""Verify strict paper-checkpoint loading and optional RGB/SAR CPU forward passes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint", required=True)
    cli.add_argument("--forward", action="store_true")
    cli.add_argument("--report", type=Path, required=True)
    options = cli.parse_args()
    import torch
    from args import get_parser
    from lib import segmentation
    from tasks.code.model import RemoteSAM, build_tokenizer, embed_sentences

    torch.set_num_threads(4)
    started = time.monotonic()
    config = json.loads((ROOT / "configs/paper_model.json").read_text())
    args = get_parser().parse_args([])
    for key, value in config.items():
        setattr(args, key, value)
    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    saved = checkpoint.get("args", {})
    saved = vars(saved) if hasattr(saved, "__dict__") else saved
    for key in ("model", "vision_backbone", "text_encoder", "img_size", "max_tokens", "modal_lora_stages",
                "modal_lora_r", "modal_lora_alpha", "modal_lora_dropout", "no_text_adapter"):
        if key in saved and str(saved[key]) != str(config[key]):
            raise ValueError(f"Checkpoint mismatch for {key}: saved={saved[key]}, config={config[key]}")
    model = segmentation.__dict__[args.model](pretrained="", args=args).eval()
    state = checkpoint.get("model", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    stages = sorted({key.split("stages.")[1].split(".")[0] for key in state if ".lora_" in key and "stages." in key})
    if stages != ["0", "1"]:
        raise ValueError(f"Unexpected checkpoint LoRA stages: {stages}")
    operations = ("referring_seg", "semantic_seg", "visual_grounding", "detection", "multi_label_cls", "multi_class_cls", "captioning", "counting")
    for name in operations:
        if not callable(getattr(RemoteSAM, name, None)):
            raise ValueError(f"Missing task operation: {name}")
    forwards = []
    if options.forward:
        tokenizer = build_tokenizer(args)
        inputs, attention = embed_sentences(["ship in the image"], tokenizer, args.max_tokens)
        inputs, attention = inputs[0], attention[0]
        image = torch.zeros((1, 3, args.img_size, args.img_size), dtype=torch.float32)
        with torch.no_grad():
            for modal in (0, 1):
                print(f"CPU forward for {'SAR' if modal else 'RGB'}...", flush=True)
                output = model(image, inputs, attention, modal=torch.tensor([modal]))
                if not torch.isfinite(output).all() or tuple(output.shape) != (1, 2, args.img_size, args.img_size):
                    raise ValueError(f"Invalid output for modality {modal}: {output.shape}")
                forwards.append({"modal": "sar" if modal else "rgb", "shape": list(output.shape), "finite": True})
    report = {"strict_checkpoint_load": True, "missing_keys": [], "unexpected_keys": [], "lora_stages": stages,
              "task_operations_present": list(operations), "parameter_count": sum(p.numel() for p in model.parameters()),
              "synthetic_cpu_forwards": forwards, "elapsed_seconds": time.monotonic() - started,
              "scope": "Synthetic forward / architecture check, not a benchmark metric or numerical parity test."}
    options.report.parent.mkdir(parents=True, exist_ok=True)
    options.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
