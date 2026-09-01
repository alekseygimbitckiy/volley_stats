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
    parser.add_argument(
        "--additional-dataset",
        action="append",
        default=[],
        help="Additional pose CSV. Use --additional-source-prefix to select derived image rows from it.",
    )
    parser.add_argument(
        "--additional-source-prefix",
        action="append",
        default=[],
        help="Filename prefix selected from each additional dataset; empty/manual rows are not copied.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT)))
    parser.add_argument("--label-column", default="action_label")
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=[],
        help="Label to remove before training. May be repeated.",
    )
    parser.add_argument(
        "--include-source-prefix",
        action="append",
        default=[],
        help=(
            "Keep derived image rows whose source-image basename starts with this value. "
            "May be repeated; rows with an empty source-image value (the manual pose data) are retained."
        ),
    )
    parser.add_argument("--source-image-column", default="source_image")
    parser.add_argument("--landmarks", choices=["main", "all"], default="main")
    parser.add_argument("--include-visibility", action="store_true")
    parser.add_argument("--include-z", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        choices=["row", "group"],
        default="row",
        help="Use group to keep correlated frames/augmentations together during evaluation.",
    )
    parser.add_argument("--group-column", default="source_group")
    parser.add_argument("--kernel", choices=["rbf", "linear", "poly", "sigmoid"], default="rbf")
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument(
        "--refit-full",
        action="store_true",
        help="After held-out evaluation, refit the saved deployment model on every usable sample.",
    )
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
    rows, additional_dataset_metadata = append_additional_dataset_rows(
        rows,
        dataset_paths=args.additional_dataset,
        prefixes=args.additional_source_prefix,
        source_image_column=args.source_image_column,
    )
    rows, source_filter_metadata = filter_rows_by_source_prefix(
        rows,
        prefixes=args.include_source_prefix,
        source_image_column=args.source_image_column,
    )
    rows, label_filter_metadata = filter_rows_by_excluded_labels(
        rows,
        label_column=args.label_column,
        excluded_labels=args.exclude_label,
    )
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

    train_indices, test_indices, split_metadata = split_indices(rows, y, args, deps)
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

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
        "source_filter": source_filter_metadata,
        "label_filter": label_filter_metadata,
        "additional_datasets": additional_dataset_metadata,
        "class_counts_total": dict(sorted(Counter(labels).items())),
        "class_counts_train": dict(sorted(Counter(y_train).items())),
        "class_counts_test": dict(sorted(Counter(y_test).items())),
        "split": {
            "test_size": args.test_size,
            "random_state": args.random_state,
            **split_metadata,
        },
        "svm": {
            "kernel": args.kernel,
            "C": args.c,
            "gamma": args.gamma,
            "class_weight": args.class_weight,
            "refit_full": args.refit_full,
            "saved_model_samples": len(rows) if args.refit_full else int(len(y_train)),
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

    if args.refit_full:
        print(f"Refitting deployment model on all {len(rows)} samples...")
        model.fit(X, y)
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
        from sklearn.model_selection import StratifiedGroupKFold, train_test_split  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        from sklearn.svm import SVC  # type: ignore
    except ModuleNotFoundError:
        accuracy_score = balanced_accuracy_score = classification_report = confusion_matrix = None
        StratifiedGroupKFold = train_test_split = Pipeline = StandardScaler = SVC = None
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
        "StratifiedGroupKFold": StratifiedGroupKFold,
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


def filter_rows_by_source_prefix(
    rows: list[dict[str, str]],
    prefixes: list[str],
    source_image_column: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    normalized = tuple(dict.fromkeys(value.strip() for value in prefixes if value.strip()))
    if not normalized:
        return rows, {"enabled": False, "rows_before": len(rows), "rows_after": len(rows)}
    if rows and source_image_column not in rows[0]:
        raise SystemExit(
            f"Source prefix filter requested, but dataset has no {source_image_column!r} column."
        )

    kept = []
    manual_rows = 0
    matched_source_rows = 0
    for row in rows:
        source_image = (row.get(source_image_column) or "").strip()
        if not source_image:
            kept.append(row)
            manual_rows += 1
            continue
        basename = Path(source_image).name
        if basename.startswith(normalized):
            kept.append(row)
            matched_source_rows += 1
    return kept, {
        "enabled": True,
        "source_image_column": source_image_column,
        "prefixes": list(normalized),
        "rows_before": len(rows),
        "rows_after": len(kept),
        "manual_rows_retained": manual_rows,
        "matched_source_rows": matched_source_rows,
    }


def filter_rows_by_excluded_labels(
    rows: list[dict[str, str]],
    label_column: str,
    excluded_labels: list[str],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    excluded = tuple(dict.fromkeys(value.strip() for value in excluded_labels if value.strip()))
    if not excluded:
        return rows, {"enabled": False, "rows_before": len(rows), "rows_after": len(rows)}
    present = {(row.get(label_column) or "").strip() for row in rows}
    unknown = sorted(set(excluded) - present)
    if unknown:
        raise SystemExit(f"Cannot exclude labels absent from the dataset: {', '.join(unknown)}")
    kept = [row for row in rows if (row.get(label_column) or "").strip() not in excluded]
    return kept, {
        "enabled": True,
        "excluded_labels": list(excluded),
        "rows_before": len(rows),
        "rows_after": len(kept),
        "rows_removed": len(rows) - len(kept),
    }


def append_additional_dataset_rows(
    primary_rows: list[dict[str, str]],
    dataset_paths: list[str],
    prefixes: list[str],
    source_image_column: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    if not dataset_paths:
        return primary_rows, []
    normalized = tuple(dict.fromkeys(value.strip() for value in prefixes if value.strip()))
    if not normalized:
        raise SystemExit("--additional-dataset requires at least one --additional-source-prefix")

    combined = list(primary_rows)
    seen_sample_ids = {(row.get("sample_id") or "").strip() for row in primary_rows}
    metadata = []
    for raw_path in dataset_paths:
        path = resolve_path(raw_path)
        candidate_rows = read_rows(path)
        if candidate_rows and source_image_column not in candidate_rows[0]:
            raise SystemExit(f"Additional dataset {path} has no {source_image_column!r} column")
        selected = []
        duplicate_samples = 0
        for row in candidate_rows:
            source_image = (row.get(source_image_column) or "").strip()
            if not source_image or not Path(source_image).name.startswith(normalized):
                continue
            sample_id = (row.get("sample_id") or "").strip()
            if sample_id and sample_id in seen_sample_ids:
                duplicate_samples += 1
                continue
            selected.append(row)
            if sample_id:
                seen_sample_ids.add(sample_id)
        combined.extend(selected)
        metadata.append(
            {
                "dataset": str(path),
                "prefixes": list(normalized),
                "rows_read": len(candidate_rows),
                "rows_added": len(selected),
                "duplicate_samples_skipped": duplicate_samples,
            }
        )
    return combined, metadata


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


def split_indices(rows: list[dict[str, str]], y: Any, args: argparse.Namespace, deps: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    np = deps["np"]
    indices = np.arange(len(rows))
    if args.split_mode == "row":
        stratify = y if can_stratify(y, args.test_size) else None
        train_indices, test_indices = deps["train_test_split"](
            indices,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify,
        )
        return train_indices, test_indices, {"mode": "row", "stratified": stratify is not None}

    groups = np.array([(row.get(args.group_column) or "").strip() for row in rows])
    missing = int(np.count_nonzero(groups == ""))
    if missing:
        raise SystemExit(
            f"Group split requested, but {missing} rows have no {args.group_column!r} value. "
            "Rebuild the combined dataset or choose --split-mode row."
        )
    n_splits = max(2, int(round(1.0 / args.test_size)))
    unique_groups = len(set(groups.tolist()))
    if unique_groups < n_splits:
        raise SystemExit(f"Need at least {n_splits} unique groups; found {unique_groups}")
    splitter = deps["StratifiedGroupKFold"](
        n_splits=n_splits,
        shuffle=True,
        random_state=args.random_state,
    )
    train_indices, test_indices = next(splitter.split(indices, y, groups))
    train_labels = set(y[train_indices].tolist())
    test_labels = set(y[test_indices].tolist())
    expected_labels = set(y.tolist())
    if train_labels != expected_labels or test_labels != expected_labels:
        raise SystemExit(
            "Group split did not place every class in both partitions. "
            f"train missing={sorted(expected_labels - train_labels)}, "
            f"test missing={sorted(expected_labels - test_labels)}. "
            "Try another --random-state."
        )
    return train_indices, test_indices, {
        "mode": "group",
        "stratified": True,
        "group_column": args.group_column,
        "groups_total": unique_groups,
        "groups_train": len(set(groups[train_indices].tolist())),
        "groups_test": len(set(groups[test_indices].tolist())),
        "effective_folds": n_splits,
    }


if __name__ == "__main__":
    raise SystemExit(main())
