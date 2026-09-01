#!/usr/bin/env python3
"""Create a filename-filtered, letterboxed copy of a YOLO detection dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="YOLO dataset containing data.yaml and split directories.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--filename-prefix", action="append", required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument("--pad-color", default="114,114,114", help="B,G,R letterbox color.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    input_root = resolve_path(args.input)
    output_root = resolve_path(args.output)
    prefixes = tuple(dict.fromkeys(value.strip() for value in args.filename_prefix if value.strip()))
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    pad_color = parse_color(args.pad_color)
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")
    if not (input_root / "data.yaml").is_file():
        raise SystemExit(f"YOLO data.yaml not found: {input_root / 'data.yaml'}")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output is not empty: {output_root}\nPass --overwrite to replace matching files.")

    cv2, np = import_dependencies()
    summary: dict[str, Any] = {
        "input": str(input_root),
        "output": str(output_root),
        "filename_prefixes": list(prefixes),
        "target_size": {"width": args.width, "height": args.height},
        "resize_mode": "letterbox",
        "splits": {},
    }
    total_images = 0
    total_labels = 0
    source_sizes: Counter[str] = Counter()

    for split in splits:
        source_images = input_root / split / "images"
        source_labels = input_root / split / "labels"
        target_images = output_root / split / "images"
        target_labels = output_root / split / "labels"
        if not source_images.is_dir() or not source_labels.is_dir():
            raise SystemExit(f"Missing source split: {split}")
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        selected = sorted(
            path
            for path in source_images.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.name.startswith(prefixes)
        )
        split_labels = 0
        missing_labels = 0
        for image_index, image_path in enumerate(selected, start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                raise SystemExit(f"Could not read image: {image_path}")
            source_height, source_width = image.shape[:2]
            source_sizes[f"{source_width}x{source_height}"] += 1
            resized, transform = letterbox_image(
                cv2,
                np,
                image,
                target_width=args.width,
                target_height=args.height,
                pad_color=pad_color,
            )
            output_image = target_images / image_path.name
            write_parameters = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality] if image_path.suffix.lower() in {".jpg", ".jpeg"} else []
            if not cv2.imwrite(str(output_image), resized, write_parameters):
                raise SystemExit(f"Could not write image: {output_image}")

            label_path = source_labels / f"{image_path.stem}.txt"
            output_label = target_labels / f"{image_path.stem}.txt"
            if label_path.exists():
                converted = convert_label_text(
                    label_path.read_text(encoding="utf-8"),
                    source_width=source_width,
                    source_height=source_height,
                    target_width=args.width,
                    target_height=args.height,
                    scale=transform["scale"],
                    pad_x=transform["pad_x"],
                    pad_y=transform["pad_y"],
                )
                output_label.write_text(converted, encoding="utf-8")
                split_labels += sum(1 for line in converted.splitlines() if line.strip())
            else:
                output_label.write_text("", encoding="utf-8")
                missing_labels += 1
            if image_index % max(1, args.progress_every) == 0:
                print(f"{split}: {image_index}/{len(selected)}")

        summary["splits"][split] = {
            "images": len(selected),
            "annotations": split_labels,
            "images_without_label_file": missing_labels,
        }
        total_images += len(selected)
        total_labels += split_labels

    write_data_yaml(input_root / "data.yaml", output_root / "data.yaml", splits)
    summary["images"] = total_images
    summary["annotations"] = total_labels
    summary["source_sizes"] = dict(sorted(source_sizes.items()))
    (output_root / "resize_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def letterbox_image(
    cv2: Any,
    np: Any,
    image: Any,
    target_width: int,
    target_height: int,
    pad_color: tuple[int, int, int],
) -> tuple[Any, dict[str, float | int]]:
    source_height, source_width = image.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = min(target_width, max(1, int(round(source_width * scale))))
    resized_height = min(target_height, max(1, int(round(source_height * scale))))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2
    canvas = np.full((target_height, target_width, 3), pad_color, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "resized_width": resized_width,
        "resized_height": resized_height,
    }


def convert_label_text(
    text: str,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> str:
    output = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        if len(values) != 5:
            raise ValueError(f"Only YOLO detection boxes are supported; line {line_number} has {len(values)} fields")
        class_id = values[0]
        cx, cy, width, height = (float(value) for value in values[1:])
        new_cx = (cx * source_width * scale + pad_x) / target_width
        new_cy = (cy * source_height * scale + pad_y) / target_height
        new_width = width * source_width * scale / target_width
        new_height = height * source_height * scale / target_height
        output.append(
            f"{class_id} {new_cx:.10f} {new_cy:.10f} {new_width:.10f} {new_height:.10f}"
        )
    return "\n".join(output) + ("\n" if output else "")


def write_data_yaml(source: Path, target: Path, splits: list[str]) -> None:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing package: pyyaml") from exc
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    path_names = {"train": "train", "valid": "val", "test": "test"}
    for split, yaml_key in path_names.items():
        if split in splits:
            payload[yaml_key] = f"{split}/images"
        else:
            payload.pop(yaml_key, None)
    target.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def parse_color(raw: str) -> tuple[int, int, int]:
    values = tuple(int(value.strip()) for value in raw.split(","))
    if len(values) != 3 or any(value < 0 or value > 255 for value in values):
        raise SystemExit("--pad-color must contain three values from 0 to 255")
    return values


def import_dependencies() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing package: opencv-python or numpy") from exc
    return cv2, np


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


if __name__ == "__main__":
    raise SystemExit(main())
