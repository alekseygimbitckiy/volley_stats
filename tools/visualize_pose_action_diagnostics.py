#!/usr/bin/env python3
"""Render representative pose-action predictions with the SVM's main landmarks."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LABELING_DIR = ROOT / "labeling"
if str(LABELING_DIR) not in sys.path:
    sys.path.insert(0, str(LABELING_DIR))

import train_pose_svm as trainer  # noqa: E402


DEFAULT_DATASET = ROOT / "data" / "processed" / "action_pose_dataset_7class" / "samples.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "action_pose_dataset_7class"
    / "diagnostics"
    / "rear_camera_main_points_misclassified.jpg"
)
TARGET_CLASSES = ("set", "serve", "spike", "block")
CONNECTIONS = (
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot_index"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET.relative_to(ROOT)))
    parser.add_argument("--additional-dataset", action="append", default=[])
    parser.add_argument("--additional-source-prefix", action="append", default=[])
    parser.add_argument("--exclude-label", action="append", default=[])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT.relative_to(ROOT)))
    parser.add_argument("--include-source-prefix", action="append", default=[])
    parser.add_argument("--classes", default=",".join(TARGET_CLASSES))
    parser.add_argument("--examples", choices=["misclassified", "correct"], default="misclassified")
    parser.add_argument(
        "--class-axis",
        choices=["ground-truth", "predicted"],
        default="ground-truth",
        help="Interpret --classes as ground-truth labels or predicted labels.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    deps = trainer.import_training_dependencies()
    cv2 = import_cv2()
    dataset_path = resolve_path(args.dataset)
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows, _ = trainer.append_additional_dataset_rows(
        rows,
        dataset_paths=args.additional_dataset,
        prefixes=args.additional_source_prefix,
        source_image_column="source_image",
    )
    rows, _ = trainer.filter_rows_by_source_prefix(rows, args.include_source_prefix, "source_image")
    rows, _ = trainer.filter_rows_by_excluded_labels(rows, "action_label", args.exclude_label)
    feature_names = trainer.build_feature_names(trainer.MAIN_LANDMARKS, include_visibility=True, include_z=False)
    rows = trainer.filter_rows(rows, "action_label", feature_names, 2)

    np = deps["np"]
    X = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype="float32")
    y = np.array([row["action_label"] for row in rows])
    split_args = argparse.Namespace(
        split_mode="group",
        group_column="source_group",
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_indices, test_indices, _ = trainer.split_indices(rows, y, split_args, deps)
    model = deps["Pipeline"](
        [
            ("scale", deps["StandardScaler"]()),
            (
                "svm",
                deps["SVC"](
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    model.fit(X[train_indices], y[train_indices])
    probabilities = model.predict_proba(X[test_indices])
    predictions = model.classes_[probabilities.argmax(axis=1)]

    requested = [value.strip() for value in args.classes.split(",") if value.strip()]
    panels = []
    for label in requested:
        dominant_false_positive_source = None
        if args.class_axis == "predicted" and args.examples == "misclassified":
            source_counts = Counter(
                str(y[row_index])
                for local_index, row_index in enumerate(test_indices)
                if predictions[local_index] == label and y[row_index] != label
            )
            if source_counts:
                dominant_false_positive_source = source_counts.most_common(1)[0][0]
        candidates = []
        for local_index, row_index in enumerate(test_indices):
            selected_label = y[row_index] if args.class_axis == "ground-truth" else predictions[local_index]
            if selected_label != label:
                continue
            is_correct = predictions[local_index] == y[row_index]
            if is_correct != (args.examples == "correct"):
                continue
            if dominant_false_positive_source is not None and y[row_index] != dominant_false_positive_source:
                continue
            predicted_index = int(probabilities[local_index].argmax())
            confidence = float(probabilities[local_index, predicted_index])
            visible = sum(float(rows[row_index][f"{name}_visibility"]) >= 0.35 for name in trainer.MAIN_LANDMARKS)
            candidates.append((visible, confidence, local_index, int(row_index)))
        if not candidates:
            raise SystemExit(f"No {args.examples} held-out example found for {label}")
        # Prefer a clearly visible pose, then a confident prediction.
        _, _, local_index, row_index = max(candidates)
        row = rows[row_index]
        print(
            f"{label}: {row['video_path']} sample={row['sample_id']} "
            f"bbox=({row['x1']},{row['y1']},{row['x2']},{row['y2']}) "
            f"gt={row['action_label']} predicted={predictions[local_index]}"
        )
        probability_by_label = {
            str(name): float(value) for name, value in zip(model.classes_, probabilities[local_index])
        }
        panel = render_panel(
            cv2,
            row,
            predicted_label=str(predictions[local_index]),
            probabilities=probability_by_label,
        )
        panels.append(panel)

    montage = make_montage(cv2, panels)
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), montage):
        raise SystemExit(f"Could not write {output_path}")
    print(f"Saved: {output_path}")
    return 0


def render_panel(cv2: Any, row: dict[str, str], predicted_label: str, probabilities: dict[str, float]) -> Any:
    image_path = resolve_path(row["video_path"])
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"Could not read {image_path}")
    h, w = frame.shape[:2]
    points = {
        name: (
            int(round(float(row[f"{name}_x"]) * w)),
            int(round(float(row[f"{name}_y"]) * h)),
            float(row[f"{name}_visibility"]),
        )
        for name in trainer.MAIN_LANDMARKS
    }
    x1, y1, x2, y2 = (float(row[name]) for name in ("x1", "y1", "x2", "y2"))
    cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), (40, 220, 70), 3)
    for first, second in CONNECTIONS:
        ax, ay, av = points[first]
        bx, by, bv = points[second]
        if min(av, bv) >= 0.35:
            cv2.line(frame, (ax, ay), (bx, by), (255, 210, 40), 3, cv2.LINE_AA)
    for index, name in enumerate(trainer.MAIN_LANDMARKS, start=1):
        x, y, visibility = points[name]
        color = (30, 70, 255) if visibility >= 0.35 else (130, 130, 130)
        cv2.circle(frame, (x, y), 6, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.putText(frame, str(index), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    bbox_w, bbox_h = x2 - x1, y2 - y1
    crop_x1 = max(0, int(x1 - 1.7 * bbox_w))
    crop_x2 = min(w, int(x2 + 1.7 * bbox_w))
    crop_y1 = max(0, int(y1 - 0.35 * bbox_h))
    crop_y2 = min(h, int(y2 + 0.25 * bbox_h))
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    panel = letterbox(cv2, crop, 900, 620)

    ground_truth = row["action_label"]
    status_color = (70, 220, 70) if predicted_label == ground_truth else (60, 90, 245)
    cv2.rectangle(panel, (0, 0), (900, 94), (20, 20, 20), -1)
    cv2.putText(
        panel,
        f"GT {ground_truth}  ->  predicted {predicted_label}",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        status_color,
        2,
        cv2.LINE_AA,
    )
    top = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:4]
    vector = "   ".join(f"{name} {value:.2f}" for name, value in top)
    cv2.putText(panel, vector, (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        Path(row["source_image"]).name[:105],
        (18, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return panel


def letterbox(cv2: Any, image: Any, width: int, height: int) -> Any:
    import numpy as np

    canvas = np.full((height, width, 3), 28, dtype=np.uint8)
    scale = min(width / max(1, image.shape[1]), height / max(1, image.shape[0]))
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def make_montage(cv2: Any, panels: list[Any]) -> Any:
    import numpy as np

    if len(panels) == 1:
        return panels[0]
    while len(panels) < 4:
        panels.append(np.full_like(panels[0], 28))
    return np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:4])])


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing package: opencv-python") from exc
    return cv2


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


if __name__ == "__main__":
    raise SystemExit(main())
