#!/usr/bin/env python3
"""Annotate a video with volleyball-ml-models action detections."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class LabeledDetection:
    frame: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a labeled video with volleyball action detections."
    )
    parser.add_argument("video", help="Path to one video clip.")
    parser.add_argument("--output-dir", default="data/processed/action_labels")
    parser.add_argument(
        "--action-model",
        default="external/volleyball-ml-models/weights/action/weights/best.pt",
        help="Path to volleyball-ml-models action YOLO weights.",
    )
    parser.add_argument("--device", default="cpu", help="Ultralytics device, for example cpu, 0, or cuda:0.")
    parser.add_argument("--action-conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--frame-stride", type=int, default=2, help="Run action model every N frames.")
    parser.add_argument("--label-hold-sec", type=float, default=2.0, help="Keep action labels visible for this long.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process the whole clip.")
    parser.add_argument(
        "--receive-labels",
        default="receive,pass",
        help="Comma-separated action class names treated as receive moments.",
    )
    parser.add_argument(
        "--ignore-labels",
        default="ball",
        help="Comma-separated classes to ignore. Default ignores the action model's ball class.",
    )
    parser.add_argument("--no-video", action="store_true", help="Only write JSON, skip annotated MP4.")
    args = parser.parse_args()

    deps = import_dependencies()
    cv2 = deps["cv2"]
    YOLO = deps["YOLO"]

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    action_model_path = resolve_model_path(Path(args.action_model))
    if not action_model_path.exists():
        raise SystemExit(
            f"Action model not found: {action_model_path}\n"
            "Download volleyball-ml-models weights first, or pass --action-model /path/to/best.pt"
        )

    print(f"Loading action model: {action_model_path}")
    action_model = YOLO(str(action_model_path))
    print(f"Action classes: {model_class_names(action_model)}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = args.max_frames if args.max_frames > 0 else frame_count

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = video_path.stem
    annotated_path = output_dir / f"{output_stem}_actions_annotated.mp4"
    json_path = output_dir / f"{output_stem}_actions.json"

    writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))

    receive_names = {name.strip().lower() for name in args.receive_labels.split(",") if name.strip()}
    ignored_names = {name.strip().lower() for name in args.ignore_labels.split(",") if name.strip()}
    hold_frames = max(1, int(round(args.label_hold_sec * fps)))
    receive_detections: list[LabeledDetection] = []
    all_action_detections: list[LabeledDetection] = []
    recent_action_detections: list[LabeledDetection] = []

    frame_idx = -1
    print(f"Video: {video_path}")
    print(f"Frames: {frame_count}, FPS: {fps:.2f}, size: {width}x{height}")
    print(f"Processing frames: {max_frames}")

    while frame_idx + 1 < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if frame_idx % max(1, args.frame_stride) == 0:
            action_result = action_model.predict(
                frame,
                conf=args.action_conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )[0]
            frame_detections = [
                detection
                for detection in parse_yolo_boxes(action_result, frame_idx)
                if detection.class_name.lower() not in ignored_names
            ]
            recent_action_detections.extend(frame_detections)
            all_action_detections.extend(frame_detections)
            receive_detections.extend(
                detection for detection in frame_detections if detection.class_name.lower() in receive_names
            )
        recent_action_detections = [
            detection for detection in recent_action_detections if frame_idx - detection.frame <= hold_frames
        ]

        if writer:
            annotated = draw_frame(
                cv2=cv2,
                frame=frame,
                frame_idx=frame_idx,
                action_detections=recent_action_detections,
                receive_names=receive_names,
                hold_frames=hold_frames,
                fps=fps,
            )
            writer.write(annotated)

        if frame_idx % 100 == 0:
            print(
                f"  frame {frame_idx}: actions={len(all_action_detections)} "
                f"receive={len(receive_detections)} recent={len(recent_action_detections)}"
            )

    cap.release()
    if writer:
        writer.release()

    result = {
        "video_id": output_stem,
        "video_path": str(video_path),
        "frame_count_processed": frame_idx + 1,
        "fps": fps,
        "action_model": str(action_model_path),
        "receive_labels": sorted(receive_names),
        "ignored_labels": sorted(ignored_names),
        "label_hold_sec": args.label_hold_sec,
        "receive_detections": [detection_to_dict(item) for item in receive_detections],
        "action_detections": [detection_to_dict(item) for item in all_action_detections],
        "receive_moments": summarize_moments(receive_detections, fps=fps, max_gap_frames=max(6, args.frame_stride * 3)),
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Receive detections: {len(receive_detections)}")
    print(f"Saved JSON: {json_path}")
    if writer:
        print(f"Saved labeled video: {annotated_path}")
    return 0


def import_dependencies() -> dict[str, Any]:
    matplotlib_cache = ROOT / ".cache" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    missing = []
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        cv2 = None
        missing.append("opencv-python")
    try:
        from ultralytics import YOLO  # type: ignore
    except ModuleNotFoundError:
        YOLO = None
        missing.append("ultralytics")
    if missing:
        raise SystemExit(
            "Missing packages: "
            + ", ".join(missing)
            + "\nInstall with: ./venv/bin/python -m pip install opencv-python ultralytics"
        )
    return {"cv2": cv2, "YOLO": YOLO}


def resolve_model_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def model_class_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", {}) or {}
    return {int(key): str(value) for key, value in dict(names).items()}


def parse_yolo_boxes(result: Any, frame_idx: int) -> list[LabeledDetection]:
    detections = []
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for idx in range(len(boxes)):
        x1, y1, x2, y2 = [float(value) for value in boxes.xyxy[idx].cpu().tolist()]
        class_id = int(boxes.cls[idx].cpu().item())
        confidence = float(boxes.conf[idx].cpu().item())
        detections.append(
            LabeledDetection(
                frame=frame_idx,
                class_name=str(names.get(class_id, f"class_{class_id}")),
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
            )
        )
    return detections


def draw_frame(
    cv2: Any,
    frame: Any,
    frame_idx: int,
    action_detections: list[LabeledDetection],
    receive_names: set[str],
    hold_frames: int,
    fps: float,
) -> Any:
    out = frame.copy()

    has_receive = False
    for detection in action_detections:
        is_receive = detection.class_name.lower() in receive_names
        if is_receive:
            has_receive = True
        color = action_color(detection.class_name)
        x1, y1, x2, y2 = [int(value) for value in detection.bbox]
        age = max(0, frame_idx - detection.frame)
        thickness = 4 if age <= 2 else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        draw_label(cv2, out, label, x1, max(28, y1 - 8), color)

    status = "RECEIVE" if has_receive else "actions"
    status_color = action_color("receive") if has_receive else (30, 30, 30)
    draw_label(cv2, out, f"frame={frame_idx} {status}", 20, 38, status_color)
    draw_action_timeline(cv2, out, action_detections, frame_idx, hold_frames, fps)
    return out


def draw_label(cv2: Any, frame: Any, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 4)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


def draw_action_timeline(
    cv2: Any,
    frame: Any,
    detections: list[LabeledDetection],
    current_frame: int,
    hold_frames: int,
    fps: float,
) -> None:
    latest_by_class: dict[str, LabeledDetection] = {}
    for detection in detections:
        key = detection.class_name.lower()
        current = latest_by_class.get(key)
        if current is None or detection.frame > current.frame or detection.confidence > current.confidence:
            latest_by_class[key] = detection

    y = 78
    for action_name in sorted(latest_by_class):
        detection = latest_by_class[action_name]
        age = current_frame - detection.frame
        if age > hold_frames:
            continue
        seconds_left = max(0.0, (hold_frames - age) / fps)
        color = action_color(action_name)
        draw_label(
            cv2,
            frame,
            f"{detection.class_name.upper()} {detection.confidence:.2f} {seconds_left:.1f}s",
            20,
            y,
            color,
        )
        y += 34


def action_color(class_name: str) -> tuple[int, int, int]:
    colors = {
        "receive": (0, 60, 255),
        "pass": (0, 60, 255),
        "serve": (0, 190, 255),
        "set": (255, 170, 0),
        "spike": (80, 0, 255),
        "block": (255, 90, 40),
        "dig": (40, 210, 80),
    }
    return colors.get(class_name.lower(), (230, 230, 230))


def detection_to_dict(detection: LabeledDetection) -> dict[str, Any]:
    x1, y1, x2, y2 = detection.bbox
    return {
        "frame": detection.frame,
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


def summarize_moments(
    detections: list[LabeledDetection],
    fps: float,
    max_gap_frames: int,
) -> list[dict[str, Any]]:
    if not detections:
        return []

    frames = sorted({item.frame for item in detections})
    groups = [[frames[0]]]
    for frame in frames[1:]:
        if frame - groups[-1][-1] <= max_gap_frames:
            groups[-1].append(frame)
        else:
            groups.append([frame])

    moments = []
    for group in groups:
        start_frame = group[0]
        end_frame = group[-1]
        moment_detections = [item for item in detections if start_frame <= item.frame <= end_frame]
        best = max(moment_detections, key=lambda item: item.confidence)
        moments.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_sec": start_frame / fps,
                "end_time_sec": end_frame / fps,
                "best_frame": best.frame,
                "best_confidence": best.confidence,
                "detections": len(moment_detections),
            }
        )
    return moments


if __name__ == "__main__":
    raise SystemExit(main())
