#!/usr/bin/env python3
"""Build a combined pose CSV used by the current action SVM.

The existing manually collected wait/receive samples are copied unchanged.
YOLO action boxes are converted to the same bbox-normalized MediaPipe landmark
features, so one SVM can be trained on all eight labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import build_action_pose_dataset as pose_data


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DATASET = ROOT / "data" / "processed" / "action_pose_dataset_batch" / "samples.csv"
DEFAULT_YOLO_DATASET = ROOT / "data" / "Volleyball Actions.v5-original.yolov11"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "action_pose_dataset_8class"

BASE_CLASSES = ("wait", "recive_bottom", "recive_top")
YOLO_CLASSES = ("block", "defense", "serve", "set", "spike")
ALL_CLASSES = BASE_CLASSES + YOLO_CLASSES
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an eight-class MediaPipe pose dataset.")
    parser.add_argument("--base-dataset", default=relative_default(DEFAULT_BASE_DATASET))
    parser.add_argument("--yolo-dataset", default=relative_default(DEFAULT_YOLO_DATASET))
    parser.add_argument("--output-dir", default=relative_default(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pose-model", default="external/mediapipe/pose_landmarker_lite.task")
    parser.add_argument("--pose-min-detection-confidence", type=float, default=0.20)
    parser.add_argument("--pose-min-visible-landmarks", type=int, default=10)
    parser.add_argument("--min-landmark-visibility", type=float, default=0.35)
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument(
        "--exclude-classes",
        default="",
        help="Comma-separated labels to omit, for example: defense",
    )
    parser.add_argument("--max-images", type=int, default=0, help="0 processes every image; useful for smoke tests.")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument(
        "--deduplicate-exact-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Process byte-identical images only once, even if they occur in multiple supplied splits.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    excluded_classes = parse_excluded_classes(args.exclude_classes)
    active_classes = tuple(label for label in ALL_CLASSES if label not in excluded_classes)

    base_path = resolve_path(args.base_dataset)
    yolo_root = resolve_path(args.yolo_dataset)
    output_dir = resolve_path(args.output_dir)
    output_path = output_dir / "samples.csv"
    summary_path = output_dir / "summary.json"
    if not base_path.is_file():
        raise SystemExit(f"Base pose dataset not found: {base_path}")
    if not yolo_root.is_dir():
        raise SystemExit(f"YOLO dataset not found: {yolo_root}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path}\nPass --overwrite to rebuild it.")

    class_names = read_yolo_class_names(yolo_root / "data.yaml")
    expected = {index: name for index, name in enumerate(YOLO_CLASSES)}
    if class_names != expected:
        raise SystemExit(f"Expected YOLO classes {expected}, found {class_names}")

    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    image_records = collect_image_records(yolo_root, splits)
    if args.max_images > 0:
        image_records = image_records[: args.max_images]

    with base_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        base_fieldnames = list(reader.fieldnames or [])
        base_rows = list(reader)
    validate_base_rows(base_rows)
    extra_fields = ["source_dataset", "source_split", "source_group", "source_image"]
    fieldnames = base_fieldnames + [name for name in extra_fields if name not in base_fieldnames]

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".csv.part")
    deps = pose_data.import_dependencies()
    cv2 = deps["cv2"]
    np = deps["np"]
    mp = pose_data.import_mediapipe()
    pose_args = SimpleNamespace(
        pose_min_detection_confidence=args.pose_min_detection_confidence,
        pose_model=str(resolve_path(args.pose_model)),
    )
    estimator = pose_data.create_pose_estimator(mp, np, pose_args)

    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    processed_images = 0
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in base_rows:
                copied = dict(row)
                copied.update(base_source_metadata(row))
                writer.writerow(copied)
                counts[copied["action_label"]] += 1

            for record in image_records:
                image_path = record["image_path"]
                if args.deduplicate_exact_images:
                    digest = file_digest(image_path)
                    if digest in seen_hashes:
                        skipped["exact_duplicate_image"] += 1
                        continue
                    seen_hashes.add(digest)

                frame = cv2.imread(str(image_path))
                if frame is None:
                    skipped["unreadable_image"] += 1
                    continue
                processed_images += 1
                for box_index, annotation in enumerate(read_yolo_labels(record["label_path"]), start=1):
                    class_id, normalized_bbox = annotation
                    label = class_names.get(class_id)
                    if label is None:
                        skipped["unknown_class"] += 1
                        continue
                    if label in excluded_classes:
                        skipped[f"excluded_{label}"] += 1
                        continue
                    bbox = normalized_xywh_to_xyxy(normalized_bbox, frame.shape[1], frame.shape[0])
                    crop, crop_origin, _ = pose_data.crop_player(frame, bbox)
                    if crop.size == 0:
                        skipped["empty_crop"] += 1
                        continue
                    landmarks_raw = estimator.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    if not landmarks_raw:
                        skipped["no_pose"] += 1
                        continue
                    landmarks = pose_data.convert_landmarks(
                        landmarks_raw,
                        crop_origin=crop_origin,
                        crop_shape=crop.shape,
                        frame_shape=frame.shape,
                        bbox=bbox,
                    )
                    visible = sum(
                        float(point["visibility"]) >= args.min_landmark_visibility for point in landmarks
                    )
                    if visible < args.pose_min_visible_landmarks:
                        skipped["weak_pose"] += 1
                        continue
                    sample = pose_data.PlayerSample(
                        sample_id=sample_id(record, box_index, label),
                        video_name=image_path.name,
                        video_path=str(image_path.relative_to(ROOT)),
                        frame_index=infer_frame_index(image_path.stem),
                        action_label=label,
                        player_index=box_index,
                        yolo_confidence=1.0,
                        bbox=bbox,
                        lower_court_intersection=0.0,
                        foot_court_score=0.0,
                        foot_points_on_court=0,
                        visible_landmarks=visible,
                        landmarks=landmarks,
                    )
                    row = pose_data.sample_to_csv_row(sample)
                    row.update(
                        {
                            "source_dataset": "volleyball_actions_yolo",
                            "source_split": record["split"],
                            "source_group": f"yolo:{source_frame_name(image_path.stem)}",
                            "source_image": str(image_path.relative_to(yolo_root)),
                        }
                    )
                    writer.writerow(row)
                    counts[label] += 1

                if processed_images % max(1, args.progress_every) == 0:
                    print(f"Processed images={processed_images}/{len(image_records)} samples={sum(counts.values())}")
    finally:
        estimator.close()

    temporary_path.replace(output_path)
    missing = [label for label in active_classes if counts[label] == 0]
    summary = {
        "output": str(output_path),
        "classes": list(active_classes),
        "excluded_classes": sorted(excluded_classes),
        "class_counts": {label: counts[label] for label in active_classes},
        "base_samples": len(base_rows),
        "candidate_yolo_images": len(image_records),
        "processed_yolo_images": processed_images,
        "skipped": dict(sorted(skipped.items())),
        "settings": vars(args),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if missing:
        raise SystemExit(f"Dataset was written, but these classes have no samples: {', '.join(missing)}")
    return 0


def relative_default(path: Path) -> str:
    return str(path.relative_to(ROOT))


def parse_excluded_classes(raw: str) -> set[str]:
    excluded = {value.strip() for value in raw.split(",") if value.strip()}
    unknown = sorted(excluded - set(ALL_CLASSES))
    if unknown:
        raise SystemExit(f"Unknown --exclude-classes labels: {', '.join(unknown)}")
    forbidden = sorted(excluded & set(BASE_CLASSES))
    if forbidden:
        raise SystemExit(
            "Excluding classes from the base pose CSV is not supported by this builder: " + ", ".join(forbidden)
        )
    return excluded


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_yolo_class_names(path: Path) -> dict[int, str]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing package: pyyaml") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = payload.get("names", {})
    if isinstance(raw, list):
        return {index: str(name) for index, name in enumerate(raw)}
    return {int(index): str(name) for index, name in raw.items()}


def collect_image_records(root: Path, splits: list[str]) -> list[dict[str, Any]]:
    records = []
    for split in splits:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise SystemExit(f"Missing YOLO split directories for {split!r} under {root}")
        images = {path.stem: path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = images.get(label_path.stem)
            if image_path is not None:
                records.append({"split": split, "image_path": image_path, "label_path": label_path})
    return records


def read_yolo_labels(path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    annotations = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        if len(values) < 5:
            raise ValueError(f"Invalid YOLO label at {path}:{line_number}")
        annotations.append((int(float(values[0])), tuple(float(value) for value in values[1:5])))
    return annotations


def normalized_xywh_to_xyxy(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    cx, cy, bw, bh = bbox
    x1 = max(0.0, min(float(width - 1), (cx - bw / 2.0) * width))
    y1 = max(0.0, min(float(height - 1), (cy - bh / 2.0) * height))
    x2 = max(x1 + 1.0, min(float(width), (cx + bw / 2.0) * width))
    y2 = max(y1 + 1.0, min(float(height), (cy + bh / 2.0) * height))
    return x1, y1, x2, y2


def source_frame_name(stem: str) -> str:
    """Collapse Roboflow augmentations of one source frame into one split group."""
    return stem.split(".rf.", 1)[0]


def infer_frame_index(stem: str) -> int:
    source = source_frame_name(stem)
    match = re.search(r"(?:-|_)(\d+)(?:_jpg|_png)?$", source)
    return int(match.group(1)) if match else 0


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sample_id(record: dict[str, Any], box_index: int, label: str) -> str:
    key = f"{record['split']}:{record['image_path'].name}:{box_index}:{label}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"yolo_{label}_{digest}"


def validate_base_rows(rows: list[dict[str, str]]) -> None:
    labels = {str(row.get("action_label", "")).strip() for row in rows}
    unexpected = sorted(labels - set(BASE_CLASSES))
    if unexpected:
        raise SystemExit(f"Base dataset contains unexpected labels: {', '.join(unexpected)}")
    missing = sorted(set(BASE_CLASSES) - labels)
    if missing:
        raise SystemExit(f"Base dataset is missing labels: {', '.join(missing)}")


def base_source_metadata(row: dict[str, str]) -> dict[str, str]:
    video_path = row.get("video_path") or row.get("video_name") or "unknown"
    try:
        frame_index = int(float(row.get("frame_index") or 0))
    except ValueError:
        frame_index = 0
    # Thirty-frame blocks stop adjacent frames from crossing the evaluation split.
    return {
        "source_dataset": "manual_receive_pose",
        "source_split": "",
        "source_group": f"manual:{video_path}:block{frame_index // 30:06d}",
        "source_image": "",
    }


if __name__ == "__main__":
    raise SystemExit(main())
