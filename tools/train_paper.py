#!/usr/bin/env python3
"""Launch the original train.py with the paper model configuration."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    for flag in ("--imageFolder", "--ref_file_path", "--instances_path"):
        if not any(value == flag or value.startswith(flag + "=") for value in extra):
            parser.error(f"{flag} must be supplied explicitly; no server-private data paths are selected")
    spec = importlib.util.spec_from_file_location("task_launcher", ROOT / "tasks/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((ROOT / "configs/paper_model.json").read_text(encoding="utf-8"))
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(args.nproc_per_node),
               "--master_port", str(args.master_port), "train.py"]
    training = json.loads((ROOT / "configs/training_defaults.json").read_text(encoding="utf-8"))
    command += module.model_arguments(config) + module.model_arguments(training) + extra
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
