#!/usr/bin/env python3
"""Exercise native argument parsers and public shell syntax without model inference."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--report", type=Path, required=True)
    options = cli.parse_args()
    registry = json.loads((ROOT / "configs/task_registry.json").read_text())
    scripts = {entry["script"] for entries in registry.values() for entry in entries.values() if "script" in entry}
    scripts.update(("test_rgb_sar.py", "examples/demo.py", "train.py", "test.py"))
    scripts.update(f"tasks/code/eval/{task}.py" for task in ("REF", "SEG", "VG", "DET", "MLC", "MCC", "CAP", "CNT"))
    failures = []
    passed = []
    for name in sorted(scripts):
        buffer = io.StringIO()
        before_argv = sys.argv
        sys.argv = [name, "--help"]
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                try:
                    runpy.run_path(str(ROOT / name), run_name="__main__")
                except SystemExit as exc:
                    if exc.code not in (None, 0):
                        raise RuntimeError(f"exit {exc.code}") from exc
            passed.append(name)
        except Exception as exc:
            failures.append({"script": name, "error": repr(exc), "output": buffer.getvalue()[-3000:]})
        finally:
            sys.argv = before_argv
    shell_passed = []
    for path in sorted((ROOT / "tasks").glob("*.sh")):
        if path.name in ("REF-contour.sh", "DET_DOTA.sh"):
            continue  # Retained inherited convenience scripts, not public canonical entry points.
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if result.returncode:
            failures.append({"script": str(path.relative_to(ROOT)), "error": result.stderr})
        else:
            shell_passed.append(str(path.relative_to(ROOT)))
    report = {"native_parsers_passed": passed, "shell_syntax_passed": shell_passed,
              "failures": failures, "scope": "No training, data loading, GPU use or model inference."}
    options.report.parent.mkdir(parents=True, exist_ok=True)
    options.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Native parsers passed: {len(passed)}; shell scripts passed: {len(shell_passed)}; failures: {len(failures)}")
    for failure in failures:
        print(json.dumps(failure), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
