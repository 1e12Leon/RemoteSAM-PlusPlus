#!/usr/bin/env python3
"""Extract SAR refs/instances from RemoteSAM_SAR_520K by original source splits.

The source datasets decide which SAR images belong to train/val/test. This
script only uses RemoteSAM_SAR_520K as an annotation/text bank: for every source
SAR image, it finds the matching 520K file name, copies or links that source
image into a new split folder, and writes a compact instances.json plus refs.p.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_SOURCE_ROOT = Path("/data/user2/SAR_Data")
DEFAULT_REMOTE_ROOT = Path("/data/user2/RemoteSAM_SAR_520K")
DEFAULT_OUTPUT_ROOT = Path("/data/user2/RemoteSAM_SAR_Source520K_BySplit")


@dataclass(frozen=True)
class SourceItem:
    dataset: str
    split: str
    image_path: Path
    lookup_stem: str
    source_sha1: str | None = None


@dataclass
class SplitResult:
    dataset: str
    split: str
    source_images: int = 0
    matched_images: int = 0
    missing_images: int = 0
    images_written: int = 0
    annotations_written: int = 0
    refs_written: int = 0
    empty_refs_skipped: int = 0
    missing_examples: list[str] | None = None

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "source_images": self.source_images,
            "matched_images": self.matched_images,
            "missing_images": self.missing_images,
            "images_written": self.images_written,
            "annotations_written": self.annotations_written,
            "refs_written": self.refs_written,
            "empty_refs_skipped": self.empty_refs_skipped,
            "missing_examples": self.missing_examples or [],
        }


class Remote520KIndex:
    def __init__(self, remote_root: Path) -> None:
        instances_path = remote_root / "instances.json"
        refs_path = remote_root / "refs(unc).p"
        if not refs_path.is_file():
            refs_path = remote_root / "refs.p"
        if not instances_path.is_file():
            raise FileNotFoundError(f"Missing 520K instances.json: {instances_path}")
        if not refs_path.is_file():
            raise FileNotFoundError(f"Missing 520K refs file: {remote_root / 'refs(unc).p'}")

        with instances_path.open("r", encoding="utf-8") as handle:
            self.instances = json.load(handle)
        with refs_path.open("rb") as handle:
            self.refs = pickle.load(handle)

        self.categories = self.instances.get("categories", [])
        self.remote_root = remote_root
        self.image_root = remote_root / "images"
        self._image_sha1_by_file: dict[str, str | None] = {}
        self.images_by_file = {img["file_name"]: img for img in self.instances.get("images", [])}
        self.annotations_by_id = {ann["id"]: ann for ann in self.instances.get("annotations", [])}
        self.files_by_stem: dict[str, list[str]] = defaultdict(list)
        for file_name in self.images_by_file:
            self.files_by_stem[Path(file_name).stem].append(file_name)

        self.refs_by_file: dict[str, list[dict]] = defaultdict(list)
        for ref in self.refs:
            self.refs_by_file[ref.get("file_name")].append(ref)

    def resolve_file(self, lookup_stem: str, preferred_suffix: str | None = None) -> str | None:
        matches = self.files_by_stem.get(lookup_stem, [])
        if not matches:
            return None
        if preferred_suffix:
            preferred_suffix = preferred_suffix.lower()
            for file_name in matches:
                if Path(file_name).suffix.lower() == preferred_suffix:
                    return file_name
        return sorted(matches)[0]

    def image_sha1(self, file_name: str) -> str | None:
        if file_name not in self._image_sha1_by_file:
            image_path = self.image_root / file_name
            self._image_sha1_by_file[file_name] = sha1_file(image_path) if image_path.is_file() else None
        return self._image_sha1_by_file[file_name]


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def iter_images(image_dir: Path) -> Iterator[Path]:
    if not image_dir.is_dir():
        return
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS:
            yield path


def find_image(directory: Path, stem_or_name: str) -> Path | None:
    raw = Path(stem_or_name)
    candidates = []
    if raw.suffix:
        candidates.append(directory / raw.name)
    stem = raw.stem
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        candidates.append(directory / f"{stem}{suffix}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def copy_or_link_image(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        raise ValueError(f"Unsupported image mode: {mode}")


def normalize_category_id(annotation: dict, ref: dict) -> None:
    if "category_id" not in annotation:
        if "categories_id" in annotation:
            annotation["category_id"] = annotation["categories_id"]
        elif "category_id" in ref:
            annotation["category_id"] = ref["category_id"]
    if "categories_id" not in annotation and "category_id" in annotation:
        annotation["categories_id"] = annotation["category_id"]


def remap_sentence_ids(ref: dict, next_sent_id: int) -> tuple[dict, int]:
    new_ref = copy.deepcopy(ref)
    new_sent_ids: list[int] = []
    for sentence in new_ref.get("sentences", []):
        sentence["sent_id"] = next_sent_id
        new_sent_ids.append(next_sent_id)
        next_sent_id += 1
    if new_sent_ids:
        new_ref["sent_ids"] = new_sent_ids
    return new_ref, next_sent_id


def ref_sentence(ref: dict) -> str:
    for sentence in ref.get("sentences", []):
        text = sentence.get("sent") or sentence.get("raw") or ""
        if str(text).strip():
            return str(text).strip()
    return ""


def sar_det_items(source_root: Path) -> Iterable[SourceItem]:
    root = source_root / "SARDet_100K"
    for split in ("train", "val", "test"):
        for image_path in iter_images(root / "Images" / split):
            yield SourceItem("SARDet_100K", split, image_path, f"{image_path.stem}_sardet100k")


def fusar_items(source_root: Path) -> Iterable[SourceItem]:
    root = source_root / "SegEarthOV2" / "FUSAR-Map"
    for split in ("train", "test"):
        for name in read_list(root / f"{split}.txt"):
            image_path = find_image(root / "sar", name)
            if image_path is not None:
                yield SourceItem("FUSAR-Map", split, image_path, f"{Path(name).stem}_FUS")


def pie_items(source_root: Path) -> Iterable[SourceItem]:
    root = source_root / "SegEarthOV2" / "PIE-RGB-SAR"
    for split in ("train", "val"):
        for name in read_list(root / f"{split}.txt"):
            image_path = find_image(root / "sar", name)
            if image_path is not None:
                yield SourceItem("PIE-RGB-SAR", split, image_path, f"{Path(name).stem}_PIE")


def yeseg_items(source_root: Path) -> Iterable[SourceItem]:
    root = source_root / "SegEarthOV2" / "YESeg-OPT-SAR"
    for split in ("train", "test"):
        for name in read_list(root / f"{split}.txt"):
            image_path = find_image(root / "sar" / Path(name).parent, Path(name).name)
            if image_path is not None:
                yield SourceItem("YESeg-OPT-SAR", split, image_path, f"{Path(name).stem}_YES", sha1_file(image_path))


def ddhrnet_items(source_root: Path) -> Iterable[SourceItem]:
    root = source_root / "SegEarthOV2" / "DDHRNet_DATA"
    configs = (
        ("korea", root / "korea" / "no_cloud", root / "korea" / "no_cloud" / "sar"),
        ("xian", root / "xian" / "no_cloud", root / "xian" / "no_cloud" / "sar"),
        ("shandong", root / "shandong", root / "shandong" / "sar"),
    )
    for region, split_root, image_dir in configs:
        for split in ("train", "val"):
            for name in read_list(split_root / f"{split}.txt"):
                image_path = find_image(image_dir, name)
                if image_path is not None:
                    yield SourceItem("DDHRNet_DATA", split, image_path, f"{Path(name).stem}_DDH_{region}")


def openearthmap_items(source_root: Path) -> Iterable[SourceItem]:
    root = source_root / "SegEarthOV2" / "OpenEarthMap_SAR"
    for split in ("train", "val", "test"):
        for image_path in iter_images(root / split / "sar_images"):
            yield SourceItem("OpenEarthMap_SAR", split, image_path, f"{image_path.stem}_OPE")


DATASET_BUILDERS: dict[str, Callable[[Path], Iterable[SourceItem]]] = {
    "SARDet_100K": sar_det_items,
    "DDHRNet_DATA": ddhrnet_items,
    "FUSAR-Map": fusar_items,
    "OpenEarthMap_SAR": openearthmap_items,
    "PIE-RGB-SAR": pie_items,
    "YESeg-OPT-SAR": yeseg_items,
}


def selected_items(source_root: Path, dataset_names: list[str]) -> Iterator[SourceItem]:
    for dataset_name in dataset_names:
        builder = DATASET_BUILDERS[dataset_name]
        yield from builder(source_root)


def build_outputs(
    items: Iterable[SourceItem],
    index: Remote520KIndex,
    output_root: Path,
    image_mode: str,
    overwrite: bool,
    clean_output: bool,
    dry_run: bool,
) -> list[SplitResult]:
    grouped: dict[tuple[str, str], list[SourceItem]] = defaultdict(list)
    for item in items:
        grouped[(item.dataset, item.split)].append(item)

    results: list[SplitResult] = []
    for (dataset, split), split_items in sorted(grouped.items()):
        result = SplitResult(dataset=dataset, split=split, missing_examples=[])
        out_dir = output_root / dataset / split
        image_dir = out_dir / "images"
        if clean_output and not dry_run and out_dir.exists():
            shutil.rmtree(out_dir)
        coco = {
            "info": {"description": f"{dataset} {split} SAR annotations extracted from RemoteSAM_SAR_520K"},
            "licenses": index.instances.get("licenses", []),
            "categories": index.categories,
            "images": [],
            "annotations": [],
        }
        refs_out: list[dict] = []
        file_to_new_image_id: dict[str, int] = {}
        next_image_id = 1
        next_ann_id = 1
        next_ref_id = 1
        next_sent_id = 1
        used_ann_ids: set[int] = set()

        for item in split_items:
            result.source_images += 1
            remote_file = index.resolve_file(item.lookup_stem, item.image_path.suffix)
            if remote_file is None:
                result.missing_images += 1
                if len(result.missing_examples or []) < 20:
                    result.missing_examples.append(f"{item.image_path} -> {item.lookup_stem}")
                continue
            if item.source_sha1 is not None and index.image_sha1(remote_file) != item.source_sha1:
                continue

            remote_refs = index.refs_by_file.get(remote_file, [])
            if not remote_refs:
                result.missing_images += 1
                if len(result.missing_examples or []) < 20:
                    result.missing_examples.append(f"{item.image_path} -> {remote_file} has no refs")
                continue

            result.matched_images += 1
            if remote_file not in file_to_new_image_id:
                src_image = index.images_by_file[remote_file]
                new_image_id = next_image_id
                next_image_id += 1
                file_to_new_image_id[remote_file] = new_image_id
                new_image = copy.deepcopy(src_image)
                new_image["id"] = new_image_id
                new_image["file_name"] = remote_file
                coco["images"].append(new_image)
                result.images_written += 1
                if not dry_run:
                    copy_or_link_image(item.image_path, image_dir / remote_file, image_mode, overwrite)

            new_image_id = file_to_new_image_id[remote_file]
            for ref in remote_refs:
                if not ref_sentence(ref):
                    result.empty_refs_skipped += 1
                    continue
                old_ann_id = ref.get("ann_id")
                if old_ann_id in used_ann_ids:
                    continue
                old_ann = index.annotations_by_id.get(old_ann_id)
                if old_ann is None:
                    continue
                used_ann_ids.add(old_ann_id)

                new_ann_id = next_ann_id
                next_ann_id += 1
                new_ann = copy.deepcopy(old_ann)
                new_ann["id"] = new_ann_id
                new_ann["image_id"] = new_image_id
                normalize_category_id(new_ann, ref)
                coco["annotations"].append(new_ann)

                new_ref, next_sent_id = remap_sentence_ids(ref, next_sent_id)
                new_ref["ref_id"] = next_ref_id
                next_ref_id += 1
                new_ref["ann_id"] = new_ann_id
                new_ref["image_id"] = new_image_id
                new_ref["file_name"] = remote_file
                new_ref["split"] = split
                new_ref["modal"] = "sar"
                if "category_id" not in new_ref and "category_id" in new_ann:
                    new_ref["category_id"] = new_ann["category_id"]
                refs_out.append(new_ref)

        result.annotations_written = len(coco["annotations"])
        result.refs_written = len(refs_out)
        results.append(result)

        if dry_run:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        for json_name in ("instances.json", "annotations.json"):
            with (out_dir / json_name).open("w", encoding="utf-8") as handle:
                json.dump(coco, handle, ensure_ascii=False)
        with (out_dir / "refs.p").open("wb") as handle:
            pickle.dump(refs_out, handle)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--remote_root", type=Path, default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_BUILDERS),
        default=sorted(DATASET_BUILDERS),
        help="Datasets to extract. Default: all six SAR datasets.",
    )
    parser.add_argument("--image_mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clean_output", action="store_true", help="Remove selected output split dirs before writing.")
    parser.add_argument("--dry-run", action="store_true", help="Only print matching statistics; do not write output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = Remote520KIndex(args.remote_root)
    items = selected_items(args.source_root, args.datasets)
    results = build_outputs(
        items=items,
        index=index,
        output_root=args.output_root,
        image_mode=args.image_mode,
        overwrite=args.overwrite,
        clean_output=args.clean_output,
        dry_run=args.dry_run,
    )

    summary = [result.as_dict() for result in results]
    for row in summary:
        print(
            "{dataset}/{split}: source={source_images} matched={matched_images} "
            "missing={missing_images} images={images_written} anns={annotations_written} "
            "refs={refs_written} empty_refs_skipped={empty_refs_skipped}".format(**row)
        )
        if row["missing_examples"]:
            print("  missing examples:")
            for example in row["missing_examples"][:5]:
                print(f"    {example}")

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        with (args.output_root / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(f"Saved summary: {args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
