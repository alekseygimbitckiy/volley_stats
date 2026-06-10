#!/usr/bin/env python3
"""Train and evaluate an SVM on the batch pose classification dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "processed" / "action_pose_dataset_batch" / "samples.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "action_pose_dataset_batch" / "svm_model"

MAIN_LANDMARKS = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_foot_index",
    "right_foot_index",
]

ALL_LANDMARKS = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit and evaluate an SVM pose classifier.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET.relative_to(ROOT)))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT)))
    parser.add_argument("--label-column", default="action_label")
    parser.add_argument("--landmarks", choices=["main", "all"], default="main")
    parser.add_argument("--include-visibility", action="store_true")
    parser.add_argument("--include-z", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--kernel", choices=["rbf", "linear", "poly", "sigmoid"], default="rbf")
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument(
        "--min-class-samples",
        type=int,
        default=2,
        help="Drop labels with fewer than this many samples before splitting.",
    )
    args = parser.parse_args()

    deps = import_training_dependencies()
    dataset_path = resolve_path(args.dataset)
    output_dir = resolve_path(args.output_dir)

    rows = read_rows(dataset_path)
    landmarks = MAIN_LANDMARKS if args.landmarks == "main" else ALL_LANDMARKS
    feature_names = build_feature_names(landmarks, include_visibility=args.include_visibility, include_z=args.include_z)
    rows = filter_rows(rows, args.label_column, feature_names, args.min_class_samples)

    if not rows:
        raise SystemExit("No usable rows after filtering.")

    labels = [row[args.label_column] for row in rows]
    if len(set(labels)) < 2:
        raise SystemExit(f"Need at least 2 labels to train SVM. Found: {sorted(set(labels))}")

    np = deps["np"]
    X = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype="float32")
    y = np.array(labels)

    train_test_split = deps["train_test_split"]
    stratify = y if can_stratify(y, args.test_size) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    Pipeline = deps["Pipeline"]
    StandardScaler = deps["StandardScaler"]
    SVC = deps["SVC"]
    classification_report = deps["classification_report"]
    confusion_matrix = deps["confusion_matrix"]
    accuracy_score = deps["accuracy_score"]
    balanced_accuracy_score = deps["balanced_accuracy_score"]
    joblib = deps["joblib"]

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel=args.kernel,
                    C=args.c,
                    gamma=args.gamma,
                    class_weight=None if args.class_weight == "none" else args.class_weight,
                    probability=True,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    labels_sorted = sorted(set(y_train) | set(y_test))
    metrics = {
        "dataset": str(dataset_path),
        "samples_total": len(rows),
        "samples_train": int(len(y_train)),
        "samples_test": int(len(y_test)),
        "features": feature_names,
        "feature_count": len(feature_names),
        "label_column": args.label_column,
        "class_counts_total": dict(sorted(Counter(labels).items())),
        "class_counts_train": dict(sorted(Counter(y_train).items())),
        "class_counts_test": dict(sorted(Counter(y_test).items())),
        "split": {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "stratified": stratify is not None,
        },
        "svm": {
            "kernel": args.kernel,
            "C": args.c,
            "gamma": args.gamma,
            "class_weight": args.class_weight,
        },
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, labels=labels_sorted, output_dict=True, zero_division=0),
        "confusion_matrix": {
            "labels": labels_sorted,
            "matrix": confusion_matrix(y_test, y_pred, labels=labels_sorted).tolist(),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "pose_svm.joblib"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "classification_report.txt"
    features_path = output_dir / "features.json"

    joblib.dump({"model": model, "feature_names": feature_names, "labels": labels_sorted, "args": vars(args)}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    features_path.write_text(json.dumps(feature_names, indent=2) + "\n", encoding="utf-8")
    report_text = classification_report(y_test, y_pred, labels=labels_sorted, zero_division=0)
    report_path.write_text(report_text + "\n", encoding="utf-8")

    print(f"Samples: total={len(rows)} train={len(y_train)} test={len(y_test)}")
    print(f"Features: {len(feature_names)}")
    print(f"Classes: {dict(sorted(Counter(labels).items()))}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(report_text)
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    return 0


def import_training_dependencies() -> dict[str, Any]:
    missing = []
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError:
        np = None
        missing.append("numpy")
    try:
        import joblib  # type: ignore
    except ModuleNotFoundError:
        joblib = None
        missing.append("joblib")
    try:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        from sklearn.svm import SVC  # type: ignore
    except ModuleNotFoundError:
        accuracy_score = balanced_accuracy_score = classification_report = confusion_matrix = None
        train_test_split = Pipeline = StandardScaler = SVC = None
        missing.append("scikit-learn")

    if missing:
        raise SystemExit(
            "Missing packages: "
            + ", ".join(sorted(set(missing)))
            + "\nInstall with: ./venv/bin/python -m pip install "
            + " ".join(sorted(set(missing)))
        )
    return {
        "np": np,
        "joblib": joblib,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "train_test_split": train_test_split,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "SVC": SVC,
    }


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_feature_names(landmarks: list[str], include_visibility: bool, include_z: bool) -> list[str]:
    names = []
    for landmark in landmarks:
        names.extend([f"{landmark}_bbox_x", f"{landmark}_bbox_y"])
        if include_z:
            names.append(f"{landmark}_z")
        if include_visibility:
            names.append(f"{landmark}_visibility")
    return names


def filter_rows(
    rows: list[dict[str, str]],
    label_column: str,
    feature_names: list[str],
    min_class_samples: int,
) -> list[dict[str, str]]:
    usable = []
    for row in rows:
        label = (row.get(label_column) or "").strip()
        if not label:
            continue
        if all(is_float(row.get(name, "")) for name in feature_names):
            usable.append(row)

    counts = Counter(row[label_column] for row in usable)
    return [row for row in usable if counts[row[label_column]] >= min_class_samples]


def is_float(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def can_stratify(y: Any, test_size: float) -> bool:
    counts = Counter(y)
    if min(counts.values()) < 2:
        return False
    n_classes = len(counts)
    n_samples = len(y)
    n_test = max(1, int(round(n_samples * test_size)))
    n_train = n_samples - n_test
    return n_test >= n_classes and n_train >= n_classes


if __name__ == "__main__":
    raise SystemExit(main())
