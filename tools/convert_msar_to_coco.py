import argparse
import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
XML_DIR_HINTS = ("annotation", "annotations", "label", "labels", "xml", "xmls")
IMAGE_DIR_HINTS = ("image", "images", "jpegimages", "jpg", "imgs")

CLASS_ALIASES = {
    "ship": "ship",
    "ships": "ship",
    "aircraft": "aircraft",
    "aircrafts": "aircraft",
    "airplane": "aircraft",
    "plane": "aircraft",
    "bridge": "bridge",
    "bridges": "bridge",
    "oil tank": "oil tank",
    "oiltank": "oil tank",
    "oil_tank": "oil tank",
    "oil-tank": "oil tank",
    "tank": "oil tank",
    "storage tank": "oil tank",
    "storagetank": "oil tank",
    "storage_tank": "oil tank",
    "船只": "ship",
    "舰船": "ship",
    "船": "ship",
    "飞机": "aircraft",
    "航空器": "aircraft",
    "油罐": "oil tank",
    "储油罐": "oil tank",
    "桥梁": "bridge",
    "桥": "bridge",
}

CATEGORIES = [
    {"id": 0, "name": "ship"},
    {"id": 1, "name": "aircraft"},
    {"id": 2, "name": "oil tank"},
    {"id": 3, "name": "bridge"},
]


def norm_class(name):
    key = " ".join(str(name).strip().lower().replace("-", " ").replace("_", " ").split())
    compact = key.replace(" ", "")
    return CLASS_ALIASES.get(key) or CLASS_ALIASES.get(compact)


def find_files(root, suffixes):
    root = Path(root)
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]


def score_dir(path, hints):
    parts = [part.lower() for part in path.parts]
    return sum(any(hint in part for hint in hints) for part in parts)


def build_image_index(image_files):
    by_stem = {}
    by_name = {}
    for path in image_files:
        by_name[path.name.lower()] = path
        by_stem.setdefault(path.stem.lower(), path)
    return by_name, by_stem


def text_of(node, names, default=None):
    for name in names:
        child = node.find(name)
        if child is not None and child.text is not None:
            return child.text.strip()
        child = node.find(name.lower())
        if child is not None and child.text is not None:
            return child.text.strip()
        child = node.find(name.upper())
        if child is not None and child.text is not None:
            return child.text.strip()
    return default


def float_text(node, names, default=None):
    value = text_of(node, names, default)
    if value is None:
        return None
    return float(value)


def find_objects(root):
    objects = list(root.findall(".//object"))
    objects.extend(root.findall(".//Object"))
    objects.extend(root.findall(".//target"))
    objects.extend(root.findall(".//Target"))
    if objects:
        return objects
    return [root]


def parse_box(obj):
    box_node = obj.find("bndbox") or obj.find("BndBox") or obj.find("box") or obj.find("Box")
    node = box_node if box_node is not None else obj
    xmin = float_text(node, ("xmin", "x_min", "Xmin", "XMin", "left", "Left"))
    ymin = float_text(node, ("ymin", "y_min", "Ymin", "YMin", "top", "Top"))
    xmax = float_text(node, ("xmax", "x_max", "Xmax", "XMax", "right", "Right"))
    ymax = float_text(node, ("ymax", "y_max", "Ymax", "YMax", "bottom", "Bottom"))
    if None in (xmin, ymin, xmax, ymax):
        return None
    x1, x2 = sorted((xmin, xmax))
    y1, y2 = sorted((ymin, ymax))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def image_for_xml(xml_path, root, by_name, by_stem):
    filename = text_of(root, ("filename", "file_name", "Filename", "FileName"))
    if filename:
        name_key = Path(filename).name.lower()
        if name_key in by_name:
            return by_name[name_key]
        stem_key = Path(filename).stem.lower()
        if stem_key in by_stem:
            return by_stem[stem_key]
    return by_stem.get(xml_path.stem.lower())


def convert_subset(xml_files, image_files, output_json, image_root=None):
    by_name, by_stem = build_image_index(image_files)
    cat_to_id = {item["name"]: item["id"] for item in CATEGORIES}
    images = []
    annotations = []
    unknown_classes = {}
    missing_images = []
    image_id_by_path = {}
    ann_id = 0

    base = Path(image_root) if image_root else common_image_root(image_files)

    for xml_path in sorted(xml_files):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        image_path = image_for_xml(xml_path, root, by_name, by_stem)
        if image_path is None:
            missing_images.append(str(xml_path))
            continue
        if image_path not in image_id_by_path:
            image_id = len(images)
            image_id_by_path[image_path] = image_id
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception:
                width = int(float(text_of(root, ("width", "Width"), 0) or 0))
                height = int(float(text_of(root, ("height", "Height"), 0) or 0))
            file_name = os.path.relpath(image_path, base)
            images.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                }
            )
        image_id = image_id_by_path[image_path]

        for obj in find_objects(root):
            name = text_of(obj, ("name", "class", "category", "targetType", "type", "Name", "Class"))
            class_name = norm_class(name)
            if class_name is None:
                if name:
                    unknown_classes[name] = unknown_classes.get(name, 0) + 1
                continue
            bbox = parse_box(obj)
            if bbox is None:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_to_id[class_name],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "ignore": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(coco, handle)
    return {
        "images": len(images),
        "annotations": len(annotations),
        "unknown_classes": unknown_classes,
        "missing_images": len(missing_images),
        "image_root": str(base),
    }


def common_image_root(image_files):
    scored = {}
    for path in image_files:
        parent = path.parent
        scored[parent] = scored.get(parent, 0) + 1
    if not scored:
        return Path(".")
    best = max(scored, key=lambda p: (score_dir(p, IMAGE_DIR_HINTS), scored[p]))
    return best


def split_xmls(xml_files, val_ratio, seed):
    rng = random.Random(seed)
    shuffled = sorted(xml_files)
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_ratio))
    val = set(shuffled[:val_count])
    train = [path for path in shuffled if path not in val]
    return train, sorted(val)


def main(args):
    root = Path(args.msar_root).expanduser().resolve()
    image_files = find_files(root, IMAGE_EXTS)
    xml_files = find_files(root, {".xml"})

    if not image_files:
        raise FileNotFoundError(f"No images found under {root}")
    if not xml_files:
        raise FileNotFoundError(f"No XML annotations found under {root}")

    image_files = sorted(image_files, key=lambda p: (score_dir(p.parent, IMAGE_DIR_HINTS), str(p)), reverse=True)
    xml_files = sorted(xml_files, key=lambda p: (score_dir(p.parent, XML_DIR_HINTS), str(p)), reverse=True)
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else root

    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.split_mode == "all":
        subsets = {"test": xml_files}
    else:
        train, val = split_xmls(xml_files, args.val_ratio, args.seed)
        subsets = {"train": train, "test": val}

    for split, split_xmls_ in subsets.items():
        output_json = output_dir / f"{split}.json"
        info = convert_subset(split_xmls_, image_files, output_json, image_root=image_root)
        print(
            f"{split}: images={info['images']} annotations={info['annotations']} "
            f"missing_images={info['missing_images']} image_root={info['image_root']} json={output_json}"
        )
        if info["unknown_classes"]:
            print(f"{split}: unknown classes {info['unknown_classes']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MSAR-1.0 XML annotations to COCO JSON.")
    parser.add_argument("--msar_root", required=True, help="Extracted MSAR-1.0 root directory.")
    parser.add_argument("--output_dir", default="/data/dishimin/liangyao/MSAR-1.0/Annotations_COCO")
    parser.add_argument("--image_root", default="", help="Optional image root passed to DET --imageFolder.")
    parser.add_argument("--split_mode", default="all", choices=["all", "train_val"])
    parser.add_argument("--val_ratio", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    main(parser.parse_args())
