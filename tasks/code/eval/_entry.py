"""Shared public task interface; numerical evaluation remains in native adapters."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tasks.run import main


def evaluate(task):
    raise SystemExit(main(task=task))
