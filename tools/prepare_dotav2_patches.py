#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
Image.MAX_IMAGE_PIXELS = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split DOTAv2 images and OBB labels into fixed-size patches for MLC evaluation."
    )
    parser.add_argument("--image-dir", default="/data/user2/zjj/DOTAv2/val/images")
    parser.add_argument("--label-dir", default="/data/user2/zjj/DOTAv2/val/labelTxt-v2.0")
    parser.add_argument("--out-dir", default="/data/user2/zjj/DOTAv2_patches/val")
    parser.add_argument("--subsize", type=int, default=1024)
    parser.add_argument("--gap", type=int, default=200)
    parser.add_argument("--min-inter-area", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def starts_for_length(length, subsize, gap):
    if length <= subsize:
        return [0]

    stride = subsize - gap
    starts = list(range(0, length - subsize + 1, stride))
    last = length - subsize
    if starts[-1] != last:
        starts.append(last)
    return starts


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(idx + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def intersect_segment(p1, p2, edge, value):
    x1, y1 = p1
    x2, y2 = p2
    if edge in ("left", "right"):
        if x1 == x2:
            return value, y1
        t = (value - x1) / (x2 - x1)
        return value, y1 + t * (y2 - y1)

    if y1 == y2:
        return x1, value
    t = (value - y1) / (y2 - y1)
    return x1 + t * (x2 - x1), value


def inside(point, edge, value):
    x, y = point
    if edge == "left":
        return x >= value
    if edge == "right":
        return x <= value
    if edge == "top":
        return y >= value
    if edge == "bottom":
        return y <= value
    raise ValueError(edge)


def clip_polygon_to_box(poly, subsize):
    output = list(poly)
    for edge, value in (
        ("left", 0.0),
        ("right", float(subsize)),
        ("top", 0.0),
        ("bottom", float(subsize)),
    ):
        if not output:
            return []
        clipped = []
        previous = output[-1]
        previous_inside = inside(previous, edge, value)
        for current in output:
            current_inside = inside(current, edge, value)
            if current_inside:
                if not previous_inside:
                    clipped.append(intersect_segment(previous, current, edge, value))
                clipped.append(current)
            elif previous_inside:
                clipped.append(intersect_segment(previous, current, edge, value))
            previous = current
            previous_inside = current_inside
        output = clipped
    return output


def read_labels(label_file):
    objects = []
    if not label_file.is_file():
        return objects

    with label_file.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                coords = [float(value) for value in parts[:8]]
            except ValueError:
                continue
            poly = [(coords[idx], coords[idx + 1]) for idx in range(0, 8, 2)]
            if polygon_area(poly) <= 0:
                continue
            objects.append((poly, parts[8:]))
    return objects


def format_number(value):
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}"


def write_patch_label(label_path, objects, x0, y0, subsize, min_inter_area):
    lines = []
    for poly, tail in objects:
        shifted = [(x - x0, y - y0) for x, y in poly]
        clipped = clip_polygon_to_box(shifted, subsize)
        if polygon_area(clipped) < min_inter_area:
            continue

        coords = []
        for x, y in shifted:
            coords.extend((x, y))
        lines.append(" ".join(format_number(v) for v in coords) + " " + " ".join(tail))

    with label_path.open("w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
    return len(lines)


def collect_images(image_dir):
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)
    out_image_dir = out_dir / "images"
    out_label_dir = out_dir / "annfiles"

    if args.subsize <= 0:
        raise ValueError("--subsize must be positive")
    if args.gap < 0 or args.gap >= args.subsize:
        raise ValueError("--gap must be in [0, subsize)")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No images found under {image_dir}")

    source_with_labels = 0
    patch_count = 0
    labeled_patch_count = 0
    object_refs = 0
    class_counts = {}

    for image_idx, image_path in enumerate(images, start=1):
        label_file = label_dir / f"{image_path.stem}.txt"
        objects = read_labels(label_file)
        if objects:
            source_with_labels += 1

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            xs = starts_for_length(width, args.subsize, args.gap)
            ys = starts_for_length(height, args.subsize, args.gap)

            for y0 in ys:
                for x0 in xs:
                    patch_name = f"{image_path.stem}__{x0}___{y0}.png"
                    patch_image_path = out_image_dir / patch_name
                    patch_label_path = out_label_dir / f"{Path(patch_name).stem}.txt"

                    if args.overwrite or not patch_image_path.exists():
                        crop = image.crop((x0, y0, min(x0 + args.subsize, width), min(y0 + args.subsize, height)))
                        patch = Image.new("RGB", (args.subsize, args.subsize))
                        patch.paste(crop, (0, 0))
                        patch.save(patch_image_path)

                    if args.overwrite or not patch_label_path.exists():
                        written = write_patch_label(
                            patch_label_path,
                            objects,
                            x0,
                            y0,
                            args.subsize,
                            args.min_inter_area,
                        )
                    else:
                        with patch_label_path.open("r") as f:
                            written = sum(1 for line in f if len(line.split()) >= 9)

                    patch_count += 1
                    if written:
                        labeled_patch_count += 1
                        object_refs += written
                        with patch_label_path.open("r") as f:
                            for line in f:
                                parts = line.split()
                                if len(parts) >= 9:
                                    class_counts[parts[8]] = class_counts.get(parts[8], 0) + 1

        if image_idx % 25 == 0 or image_idx == len(images):
            print(f"Processed {image_idx}/{len(images)} images; patches: {patch_count}")

    manifest = {
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "out_dir": str(out_dir),
        "subsize": args.subsize,
        "gap": args.gap,
        "min_inter_area": args.min_inter_area,
        "source_images": len(images),
        "source_images_with_labels": source_with_labels,
        "patches": patch_count,
        "labeled_patches": labeled_patch_count,
        "object_refs_in_patch_labels": object_refs,
        "class_counts": dict(sorted(class_counts.items())),
    }
    manifest_path = out_dir / "patch_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
