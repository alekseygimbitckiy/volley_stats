#!/usr/bin/env python3
"""Run fast-volleyball-tracking-inference and normalize its ball.csv output."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAST_DIR = ROOT / "external" / "fast-volleyball-tracking-inference"
ONNX_SCRIPT = FAST_DIR / "src" / "inference_onnx_seq_gray_v2.py"
OPENVINO_SCRIPT = FAST_DIR / "src" / "inference_openvino_seq_gray_v2.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fast volleyball ball tracking.")
    parser.add_argument("video", help="Path to one video.")
    parser.add_argument("--output-dir", default="data/processed/fast_vball_raw")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--backend", choices=["onnx", "openvino"], default="onnx")
    parser.add_argument(
        "--model-path",
        default="external/fast-volleyball-tracking-inference/models/VballNetV1_seq9_grayscale_148_h288_w512.onnx",
        help="ONNX model path for --backend onnx.",
    )
    parser.add_argument(
        "--model-xml",
        default="external/fast-volleyball-tracking-inference/ov/VballNetV2_seq9_grayscale_ov.xml",
        help="OpenVINO XML model path for --backend openvino.",
    )
    parser.add_argument("--track-length", type=int, default=32)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--keep-video", action="store_true", help="Also write the fast repo predict.mp4.")
    args = parser.parse_args()

    video_path = resolve_project_path(args.video)
    output_root = resolve_project_path(args.output_dir)
    check_ready(args, video_path)

    command = build_command(args, video_path, output_root)
    print("Running fast volleyball tracking:")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=str(FAST_DIR), text=True)
    if completed.returncode != 0:
        return completed.returncode

    raw_csv = output_root / video_path.stem / "ball.csv"
    if not raw_csv.exists():
        print(f"Expected fast tracker CSV was not created: {raw_csv}", file=sys.stderr)
        return 1

    rows = normalize_fast_csv(raw_csv)
    normalized_csv = output_root / f"{video_path.stem}.csv"
    normalized_json = output_root / f"{video_path.stem}.json"
    write_project_csv(normalized_csv, rows)
    normalized_json.write_text(json.dumps({"ball_positions": rows}, indent=2) + "\n", encoding="utf-8")

    print(f"Fast tracker raw CSV: {raw_csv}")
    print(f"Normalized CSV: {normalized_csv}")
    print(f"Normalized JSON: {normalized_json}")
    print(f"Ball frames with coordinates: {sum(1 for row in rows if row['x'] is not None and row['y'] is not None)} / {len(rows)}")
    return 0


def build_command(args: argparse.Namespace, video_path: Path, output_root: Path) -> list[str]:
    if args.backend == "openvino":
        command = [
            args.python,
            str(OPENVINO_SCRIPT),
            "--video_path",
            str(video_path),
            "--model_xml",
            str(resolve_project_path(args.model_xml)),
            "--output_dir",
            str(output_root),
            "--track_length",
            str(args.track_length),
        ]
    else:
        command = [
            args.python,
            str(ONNX_SCRIPT),
            "--video_path",
            str(video_path),
            "--model_path",
            str(resolve_project_path(args.model_path)),
            "--output_dir",
            str(output_root),
            "--track_length",
            str(args.track_length),
        ]
    if args.confidence_threshold is not None:
        command.extend(["--confidence_threshold", str(args.confidence_threshold)])
    if not args.keep_video:
        command.append("--only_csv")
    return command


def check_ready(args: argparse.Namespace, video_path: Path) -> None:
    missing = []
    if not video_path.exists():
        missing.append(f"video not found: {video_path}")
    if not FAST_DIR.exists():
        missing.append(f"fast volleyball repo not found: {FAST_DIR}")
    script = OPENVINO_SCRIPT if args.backend == "openvino" else ONNX_SCRIPT
    if not script.exists():
        missing.append(f"inference script not found: {script}")
    model_path = resolve_project_path(args.model_xml if args.backend == "openvino" else args.model_path)
    if not model_path.exists():
        missing.append(f"model not found: {model_path}")
    if missing:
        raise SystemExit("\n".join(missing))


def normalize_fast_csv(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            visibility = int(float(raw.get("Visibility") or 0))
            x = int(float(raw.get("X") or -1))
            y = int(float(raw.get("Y") or -1))
            radius = int(float(raw.get("Radius") or 0))
            diameter = radius * 2 if radius > 0 else 0
            rows.append(
                {
                    "frame": int(float(raw.get("Frame") or 0)),
                    "x": None if visibility == 0 or x < 0 else x,
                    "y": None if visibility == 0 or y < 0 else y,
                    "width": diameter,
                    "height": diameter,
                    "confidence": 1.0 if visibility else 0.0,
                    "visibility": visibility,
                    "source": f"fast-vball-{path.parent.name}",
                }
            )
    return rows


def write_project_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "x", "y", "width", "height", "confidence", "visibility", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
