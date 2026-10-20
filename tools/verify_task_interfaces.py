#!/usr/bin/env python3
"""Check every configured case through its public task file without inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads((ROOT / "configs/task_registry.json").read_text())
    passed, blocked, errors = [], [], []
    for group, entries in registry.items():
        task, modal = group.split("_")
        for name, entry in entries.items():
            command = [sys.executable, str(ROOT / f"tasks/code/eval/{task}.py"),
                       "--modal", modal.lower(), "--dataset", name, "--resume", "/fixture/model.pth",
                       "--images", "/fixture/images", "--annotations", "/fixture/annotations",
                       "--refs", "/fixture/refs.p", "--instances", "/fixture/instances.json",
                       "--data-root", "/fixture/data", "--classes", "ship", "--dry-run"]
            result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
            case = f"{group}/{name}"
            if entry["status"] == "needs_source":
                if result.returncode and entry["reason"] in result.stderr:
                    blocked.append(case)
                else:
                    errors.append({"case": case, "output": result.stdout + result.stderr})
            elif result.returncode or "_adapters" not in result.stdout and "test/" not in result.stdout.replace("\\", "/"):
                errors.append({"case": case, "output": result.stdout + result.stderr})
            else:
                passed.append(case)
    report = {"dry_run_cases_passed": passed, "expected_missing_source_errors": blocked,
              "errors": errors, "scope": "CLI dispatch only; no model/data loading or inference."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Dispatch passed: {len(passed)}; expected missing-source errors: {len(blocked)}; errors: {len(errors)}")
    for error in errors:
        print(error, file=sys.stderr)
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
