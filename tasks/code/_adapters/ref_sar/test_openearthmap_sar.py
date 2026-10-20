from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_sar_eval import DEFAULT_EXTRACTED_ROOT, build_common_parser, evaluate_ref_split


def main() -> None:
    parser = build_common_parser("Evaluate RemoteSAM on extracted OpenEarthMap_SAR SAR refs.")
    parser.set_defaults(data_root=DEFAULT_EXTRACTED_ROOT / "OpenEarthMap_SAR", split="test")
    args = parser.parse_args()
    evaluate_ref_split(args, "OpenEarthMap_SAR")


if __name__ == "__main__":
    main()
