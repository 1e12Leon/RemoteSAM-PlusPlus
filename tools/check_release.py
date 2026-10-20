#!/usr/bin/env python3
"""Audit source preservation, syntax, launchers, and manuscript coverage without a GPU."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-source", type=Path, help="Untouched server source snapshot for comparison.")
    parser.add_argument("--recovered-source", type=Path, help="Read-only dishimin source snapshot.")
    parser.add_argument("--report", type=Path)
    options = parser.parse_args()
    errors = []
    python_files = list(ROOT.rglob("*.py"))
    for path in python_files:
        if ".git" in path.parts:
            continue
        try:
            ast.parse(source_text(path), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    registry = json.loads((ROOT / "configs/task_registry.json").read_text())
    model = json.loads((ROOT / "configs/paper_model.json").read_text())
    assert model["modal_lora_stages"] == "0,1"
    assert model["modal_lora_targets"] == "pointwise_conv1,pointwise_conv2"
    missing_sources = []
    for key, entries in registry.items():
        if not (ROOT / "tasks/code/eval" / f"{key.split('_')[0]}.py").is_file():
            errors.append(f"Missing canonical launcher: {key}")
        for name, entry in entries.items():
            if entry["status"] == "needs_source":
                missing_sources.append({"case": f"{key}/{name}", "reason": entry["reason"]})
            elif not (ROOT / entry["script"]).is_file():
                errors.append(f"Missing evaluator: {entry['script']}")
    for task in ("REF", "SEG", "VG", "DET", "MLC", "MCC", "CAP", "CNT"):
        for modal in ("RGB", "SAR"):
            if f"{task}_{modal}" not in registry:
                errors.append(f"Missing task/modality pair: {task}/{modal}")

    preserved = []
    changed = []
    if options.reference_source:
        reference = options.reference_source.resolve()
        paths = [Path(name) for name in ("train.py", "test.py", "transforms.py", "utils.py", "transformers_compat.py")]
        for folder in ("arc", "lib", "loss", "data", "tasks/code", "test"):
            paths += [p.relative_to(ROOT) for p in (ROOT / folder).rglob("*.py") if (reference / p.relative_to(ROOT)).is_file() and "eval" not in p.relative_to(ROOT).parts]
        for path in (ROOT / "tasks/code/_adapters").rglob("*.py"):
            old = Path("test") / path.name if "ref_sar" in path.parts else Path("tasks/code/eval") / path.relative_to(ROOT / "tasks/code/_adapters")
            if (reference / old).is_file():
                before = source_text(reference / old).replace("tasks.code.eval.CAP", "tasks.code._adapters.CAP")
                if old == Path("test/common_sar_eval.py"):
                    before = before.replace("ROOT = Path(__file__).resolve().parents[1]", "ROOT = Path(__file__).resolve().parents[4]")
                if before != source_text(path):
                    errors.append(f"Relocated adapter implementation differs: {old}")
                else:
                    preserved.append({"path": str(path.relative_to(ROOT)), "original_path": str(old), "sha256_lf": hashlib.sha256(before.encode()).hexdigest()})
        for relative in sorted(set(paths)):
            before, after = reference / relative, ROOT / relative
            if not before.is_file():
                continue
            if source_text(before) != source_text(after):
                changed.append(str(relative))
                errors.append(f"Protected implementation differs from server: {relative}")
            else:
                preserved.append({"path": str(relative), "sha256_lf": hashlib.sha256(source_text(after).encode()).hexdigest()})

    recovered = []
    if options.recovered_source:
        mapping = {
            "lib/modal_inference.py": "lib/modal_inference.py",
            "tasks/code/model_auto.py": "tasks/code/model.py",
            "tools/convert_msar_to_coco.py": "tools/convert_msar_to_coco.py",
            "tools/summarize_satin.py": "tasks/code/eval/merge_mlc_satin_shards.py",
            "tasks/code/_adapters/datasets_mlc.py": "data/datasets/MLC.py",
            "tasks/code/_adapters/datasets_vg.py": "data/datasets/VG.py",
            "tasks/code/_adapters/MLC_ranking.py": "tasks/code/eval/MLC.py",
            "tasks/code/_adapters/VG_refs.py": "tasks/code/eval/VG.py",
        }
        for name in ("MLC_SATIN.py", "SEG_Vaihingen.py", "SEG_DeepGlobe_folder.py"):
            mapping["tasks/code/_adapters/" + name] = "tasks/code/eval/" + name
        for destination, origin in mapping.items():
            before = source_text(options.recovered_source / origin)
            if destination.endswith(("MLC_ranking.py", "VG_refs.py")):
                before = before.replace("from data.DiverseDataset import DiverseDataset", "from tasks.code._adapters.datasets_recovered import DiverseDataset")
            if destination.endswith(("MLC_ranking.py", "VG_refs.py", "MLC_SATIN.py")):
                before = before.replace("from tasks.code.model import RemoteSAM", "from tasks.code.model_auto import RemoteSAM")
            if before != source_text(ROOT / destination):
                errors.append(f"Recovered implementation differs beyond import adaptation: {destination}")
            else:
                recovered.append(destination)

    report = {
        "python_files_parsed": len(python_files), "task_modality_pairs": len(registry),
        "preserved_implementations": preserved, "unexpected_implementation_changes": changed,
        "recovered_implementations_preserved": recovered,
        "intentional_changes": ["args.py: paper LoRA stages default 0,1", "test_rgb_sar.py: explicit import of unchanged test.py"],
        "manuscript_cases_requiring_original_source": missing_sources,
        "errors": errors,
        "verification_scope": "Static/source preservation only; not a clean install or full benchmark reproduction.",
    }
    if options.report:
        options.report.parent.mkdir(parents=True, exist_ok=True)
        options.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Python files: {len(python_files)}; task/modality pairs: {len(registry)}; preserved implementations: {len(preserved)}")
    print(f"Unresolved manuscript cases: {len(missing_sources)}; implementation/syntax errors: {len(errors)}")
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
