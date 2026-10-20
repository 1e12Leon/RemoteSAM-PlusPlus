"""Public CNT evaluation for supported RGB and SAR input formats."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tasks.code.eval._entry import evaluate

if __name__ == "__main__":
    evaluate("CNT")
