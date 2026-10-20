#!/usr/bin/env python3
"""Consistent paper-configured launchers around the unchanged task evaluators."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("REF", "SEG", "VG", "DET", "MLC", "MCC", "CAP", "CNT")


def model_arguments(config: dict) -> list[str]:
    result = []
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                result.append(f"--{key}")
        else:
            result.extend((f"--{key}", str(value)))
    return result


def build_parser(task: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    if task is None:
        parser.add_argument("--task", choices=TASKS)
    else:
        parser.set_defaults(task=task)
    parser.add_argument("--modal", choices=("rgb", "sar"))
    parser.add_argument("--dataset", help="Use --list to see the supported case names.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running models.")
    parser.add_argument("--resume", default=os.environ.get("CHECKPOINT", ""))
    parser.add_argument("--images", default=os.environ.get("IMAGE_FOLDER", ""))
    parser.add_argument("--annotations", default=os.environ.get("ANNO_FOLDER", ""))
    parser.add_argument("--refs", default=os.environ.get("REF_FILE_PATH", ""))
    parser.add_argument("--instances", default=os.environ.get("INSTANCES_PATH", ""))
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", ""))
    parser.add_argument("--split", default="test")
    parser.add_argument("--classes", nargs="+", help="Existing dataset prompt names for DET; use all classes for total AP.")
    parser.add_argument("--save-path", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epoc", action="store_true", help="Enable the existing EPOC path for DET/CNT/CAP.")
    return parser


def main(task: str | None = None) -> int:
    parser = build_parser(task)
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    registry = json.loads((ROOT / "configs/task_registry.json").read_text(encoding="utf-8"))
    if args.list:
        for key, entries in registry.items():
            for name, entry in entries.items():
                print(f"{key:<8} {name:<18} {entry['status']:<14} {entry.get('script', entry.get('reason', ''))}")
        return 0
    if not all((args.task, args.modal, args.dataset)):
        parser.error("--task, --modal and --dataset are required unless --list is used")
    key = f"{args.task}_{args.modal.upper()}"
    entry = registry[key].get(args.dataset)
    if entry is None:
        parser.error(f"Unsupported case {key}/{args.dataset}; choose from {', '.join(registry[key])}")
    if entry["status"] == "needs_source":
        parser.error(f"{key}/{args.dataset}: {entry['reason']}")
    missing = [name for name in entry["needs"] if not getattr(args, name)]
    if not args.resume:
        missing.append("resume")
    if missing:
        parser.error("Required arguments: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    if entry["status"] == "supplementary":
        print("Supplementary evaluation; not a verified reproduction of a manuscript table.", flush=True)
    if entry.get("note"):
        print(entry["note"], flush=True)

    config = json.loads((ROOT / "configs/paper_model.json").read_text(encoding="utf-8"))
    base = [sys.executable, str(ROOT / entry["script"])] + model_arguments(config)
    base += ["--resume", args.resume, "--modal", args.modal, "--device", args.device,
             "--workers", str(args.workers), "--split", args.split]
    output = args.save_path or f"./result/{key}/{args.dataset}"
    base += ["--save_path", output]
    if entry.get("device_index"):
        base += ["--stats_dir", output]
    if args.images:
        base += [entry.get("images_flag", "--imageFolder"), args.images]
    if args.annotations:
        base += [entry.get("annotations_flag", "--annoFolder"), args.annotations]
    elif args.task == "MCC":
        base += ["--annoFolder", args.images]
    if args.refs:
        base += ["--ref_file_path", args.refs]
    if args.instances:
        base += ["--instances_path", args.instances]
    if entry.get("dataset"):
        base += ["--dataset", entry["dataset"]]
    if not entry["script"].startswith("test/"):
        base += ["--task", args.task]
    if args.data_root:
        root_flag = "--seg_sar_root" if entry["script"].endswith("SEG_SAR.py") else "--data_root"
        base += [root_flag, args.data_root]
    if args.epoc:
        if args.task not in ("DET", "CNT", "CAP"):
            parser.error("--epoc is meaningful only for DET/CNT/CAP")
        base.append("--EPOC")
    base += entry.get("extra", []) + extra
    if entry.get("device_index"):
        if not args.device.startswith("cuda:"):
            parser.error("This recovered evaluator requires --device cuda:N")
        base += ["--device_index", args.device.split(":", 1)[1]]
    commands = [base + ["--_class", name] for name in args.classes] if args.task == "DET" else [base]
    if args.task == "DET" and entry["script"].endswith("/DET.py"):
        commands.append([
            sys.executable, str(ROOT / "tasks/code/_adapters/DET_COCO_summary.py"),
            "--annoFolder", args.annotations, "--pred_dir", output, "--classes", *args.classes,
            "--output", "preds_all.json",
        ])
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONUNBUFFERED"] = "1"
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
