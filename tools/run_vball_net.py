#!/usr/bin/env python3
"""Run vball-net ONNX inference and normalize its CSV output for this project."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VBALL_NET_DIR = ROOT / "external" / "vball-net"
VBALL_INFERENCE = VBALL_NET_DIR / "src" / "inference_onnx.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run vball-net ball tracking.")
    parser.add_argument("video", help="Path to one video.")
    parser.add_argument(
        "--model-path",
        default="external/vball-net/vb-models/VballNetFastV1_155_h288_w512.onnx",
        help="Path to a vball-net ONNX model.",
    )
    parser.add_argument("--output-dir", default="data/processed/vball_net_raw")
    parser.add_argument("--python", default=sys.executable, help="Python executable for vball-net.")
    parser.add_argument("--track-length", type=int, default=32)
    parser.add_argument("--keep-vball-video", action="store_true", help="Also let vball-net write its predicted video.")
    args = parser.parse_args()

    video_path = resolve_project_path(args.video)
    model_path = resolve_project_path(args.model_path)
    output_root = resolve_project_path(args.output_dir)
    raw_dir = output_root / video_path.stem
    raw_dir.mkdir(parents=True, exist_ok=True)

    check_ready(video_path, model_path)

    command = [
        args.python,
        str(VBALL_INFERENCE),
        "--video_path",
        str(video_path),
        "--model_path",
        str(model_path),
        "--output_dir",
        str(raw_dir),
        "--track_length",
        str(args.track_length),
    ]
    if not args.keep_vball_video:
        command.append("--only_csv")

    print("Running vball-net:")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=str(VBALL_NET_DIR), text=True)
    if completed.returncode != 0:
        return completed.returncode

    vball_csv = raw_dir / f"{video_path.stem}_predict_ball.csv"
    if not vball_csv.exists():
        print(f"Expected vball-net CSV was not created: {vball_csv}", file=sys.stderr)
        return 1

    normalized_csv = output_root / f"{video_path.stem}.csv"
    normalized_json = output_root / f"{video_path.stem}.json"
    rows = normalize_vball_csv(vball_csv)
    write_project_csv(normalized_csv, rows)
    normalized_json.write_text(json.dumps({"ball_positions": rows}, indent=2) + "\n", encoding="utf-8")

    print(f"vball-net raw CSV: {vball_csv}")
    print(f"Normalized CSV: {normalized_csv}")
    print(f"Normalized JSON: {normalized_json}")
    print(f"Ball frames with coordinates: {sum(1 for row in rows if row['x'] is not None and row['y'] is not None)} / {len(rows)}")
    return 0


def check_ready(video_path: Path, model_path: Path) -> None:
    missing = []
    if not video_path.exists():
        missing.append(f"video not found: {video_path}")
    if not VBALL_NET_DIR.exists():
        missing.append(f"vball-net repo not found: {VBALL_NET_DIR}")
    if not VBALL_INFERENCE.exists():
        missing.append(f"vball-net inference script not found: {VBALL_INFERENCE}")
    if not model_path.exists():
        missing.append(
            f"model not found: {model_path}\n"
            "Download a pretrained ONNX model from the links in external/vball-net/README.md "
            "and place it under external/vball-net/vb-models/."
        )
    if missing:
        raise SystemExit("\n".join(missing))

    dependency_errors = []
    for module in ("cv2", "numpy", "pandas", "onnx", "onnxruntime", "tqdm"):
        try:
            __import__(module)
        except ModuleNotFoundError:
            dependency_errors.append(module)
    if dependency_errors:
        package_names = {
            "cv2": "opencv-python",
            "numpy": "numpy",
            "pandas": "pandas",
            "onnx": "onnx",
            "onnxruntime": "onnxruntime",
            "tqdm": "tqdm",
        }
        packages = " ".join(package_names[name] for name in dependency_errors)
        raise SystemExit(
            "Missing vball-net inference packages: "
            + ", ".join(dependency_errors)
            + "\nInstall with:\n"
            + f"  python3 -m pip install {packages}"
        )


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def normalize_vball_csv(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            visibility = int(float(raw.get("Visibility") or 0))
            x = int(float(raw.get("X") or -1))
            y = int(float(raw.get("Y") or -1))
            rows.append(
                {
                    "frame": int(float(raw.get("Frame") or 0)),
                    "x": None if visibility == 0 or x < 0 else x,
                    "y": None if visibility == 0 or y < 0 else y,
                    "width": int(float(raw.get("W") or 0)),
                    "height": int(float(raw.get("H") or 0)),
                    "confidence": 1.0 if visibility else 0.0,
                    "visibility": visibility,
                    "source": "vball-net",
                }
            )
    return rows


def write_project_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "x",
                "y",
                "width",
                "height",
                "confidence",
                "visibility",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
