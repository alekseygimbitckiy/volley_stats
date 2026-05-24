#!/usr/bin/env python3
"""Label likely serve-receive moments from vball-net ball trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BallPoint:
    frame: int
    x: float
    y: float
    radius: float
    confidence: float


@dataclass
class ReceiveMoment:
    frame: int
    x: float
    y: float
    score: float
    angle_change_deg: float
    speed_before: float
    speed_after: float
    vy_before: float
    vy_after: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Label receive moments from a vball-net ball track.")
    parser.add_argument("video", help="Path to one video clip.")
    parser.add_argument("--ball-track", default=None, help="Existing normalized vball-net CSV/JSON.")
    parser.add_argument("--vball-model-path", default=None, help="Run vball-net first with this ONNX model.")
    parser.add_argument("--vball-output-dir", default="data/processed/vball_net_raw")
    parser.add_argument("--output-dir", default="data/processed/receive_from_ball")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process the whole clip.")
    parser.add_argument("--trail-length", type=int, default=45)
    parser.add_argument("--near-side-y-ratio", type=float, default=0.42, help="Only consider contacts below this image ratio.")
    parser.add_argument("--analysis-window", type=int, default=4, help="Frames before/after point for velocity estimate.")
    parser.add_argument("--min-angle-change", type=float, default=55.0)
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--label-hold-sec", type=float, default=2.0)
    parser.add_argument("--max-moments", type=int, default=3)
    parser.add_argument("--no-video", action="store_true", help="Only write JSON.")
    args = parser.parse_args()

    deps = import_dependencies()
    cv2 = deps["cv2"]

    video_path = resolve_project_path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    ball_track_path = ensure_ball_track(args, video_path)
    ball_points = load_ball_track(ball_track_path)
    if not ball_points:
        raise SystemExit(f"No visible ball points found in {ball_track_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = args.max_frames if args.max_frames > 0 else frame_count

    moments = detect_receive_moments(
        points=ball_points,
        video_height=height,
        near_side_y_ratio=args.near_side_y_ratio,
        window=args.analysis_window,
        min_angle_change=args.min_angle_change,
        min_score=args.min_score,
        max_moments=args.max_moments,
    )

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = video_path.stem
    annotated_path = output_dir / f"{output_stem}_receive_from_ball_annotated.mp4"
    json_path = output_dir / f"{output_stem}_receive_from_ball.json"

    writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))

    points_by_frame = {point.frame: point for point in ball_points}
    hold_frames = max(1, int(round(args.label_hold_sec * fps)))
    ball_history: list[BallPoint] = []

    print(f"Video: {video_path}")
    print(f"Ball track: {ball_track_path}")
    print(f"Frames: {frame_count}, FPS: {fps:.2f}, size: {width}x{height}")
    print(f"Detected receive moments: {len(moments)}")
    for moment in moments:
        print(
            f"  frame {moment.frame} time={moment.frame / fps:.2f}s "
            f"score={moment.score:.2f} angle={moment.angle_change_deg:.1f}"
        )

    frame_idx = -1
    while frame_idx + 1 < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        point = points_by_frame.get(frame_idx)
        if point is not None:
            ball_history.append(point)

        if writer:
            annotated = draw_frame(
                cv2=cv2,
                frame=frame,
                frame_idx=frame_idx,
                fps=fps,
                current_ball=point,
                ball_history=ball_history,
                trail_length=args.trail_length,
                moments=moments,
                hold_frames=hold_frames,
            )
            writer.write(annotated)

        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}: ball_points={len(ball_history)}")

    cap.release()
    if writer:
        writer.release()

    result = {
        "video_id": output_stem,
        "video_path": str(video_path),
        "ball_track": str(ball_track_path),
        "frame_count_processed": frame_idx + 1,
        "fps": fps,
        "parameters": {
            "near_side_y_ratio": args.near_side_y_ratio,
            "analysis_window": args.analysis_window,
            "min_angle_change": args.min_angle_change,
            "min_score": args.min_score,
            "label_hold_sec": args.label_hold_sec,
            "max_moments": args.max_moments,
        },
        "receive_moments": [moment_to_dict(moment, fps) for moment in moments],
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

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
    if missing:
        raise SystemExit(
            "Missing packages: "
            + ", ".join(missing)
            + "\nInstall with: ./venv/bin/python -m pip install opencv-python"
        )
    return {"cv2": cv2}


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def ensure_ball_track(args: argparse.Namespace, video_path: Path) -> Path:
    if args.vball_model_path:
        command = [
            sys.executable,
            str(ROOT / "tools" / "run_vball_net.py"),
            str(video_path),
            "--model-path",
            args.vball_model_path,
            "--output-dir",
            args.vball_output_dir,
        ]
        print("Running vball-net:")
        print(" ".join(command))
        completed = subprocess.run(command, text=True)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

    if args.ball_track:
        path = resolve_project_path(args.ball_track)
        if not path.exists():
            raise SystemExit(f"Ball track file not found: {path}")
        return path

    output_dir = resolve_project_path(args.vball_output_dir)
    for candidate in (output_dir / f"{video_path.stem}.csv", output_dir / f"{video_path.stem}.json"):
        if candidate.exists():
            return candidate

    raise SystemExit(
        "No ball track found. Run with --vball-model-path external/vball-net/vb-models/VballNetFastV1_155_h288_w512.onnx "
        "or pass --ball-track data/processed/vball_net_raw/<video>.csv"
    )


def load_ball_track(path: Path) -> list[BallPoint]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("ball_positions", data if isinstance(data, list) else [])
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    points = []
    for row in rows:
        x = row.get("x", row.get("X"))
        y = row.get("y", row.get("Y"))
        if x in (None, "", "-1") or y in (None, "", "-1"):
            continue
        frame = int(float(row.get("frame", row.get("Frame", 0)) or 0))
        width = float(row.get("width", row.get("W", 10)) or 10)
        height = float(row.get("height", row.get("H", width)) or width)
        confidence = float(row.get("confidence", row.get("Confidence", 1.0)) or 0)
        points.append(
            BallPoint(
                frame=frame,
                x=float(x),
                y=float(y),
                radius=max(5.0, (width + height) / 4),
                confidence=confidence,
            )
        )
    return sorted(points, key=lambda point: point.frame)


def detect_receive_moments(
    points: list[BallPoint],
    video_height: int,
    near_side_y_ratio: float,
    window: int,
    min_angle_change: float,
    min_score: float,
    max_moments: int,
) -> list[ReceiveMoment]:
    by_frame = {point.frame: point for point in smooth_points(points)}
    frames = sorted(by_frame)
    candidates = []
    min_y = video_height * near_side_y_ratio

    for frame in frames:
        point = by_frame[frame]
        if point.y < min_y:
            continue

        before = nearest_point(by_frame, frame - window, direction=-1, max_gap=window * 2)
        after = nearest_point(by_frame, frame + window, direction=1, max_gap=window * 2)
        if before is None or after is None:
            continue
        if before.frame >= frame or after.frame <= frame:
            continue

        vx_before = (point.x - before.x) / max(1, point.frame - before.frame)
        vy_before = (point.y - before.y) / max(1, point.frame - before.frame)
        vx_after = (after.x - point.x) / max(1, after.frame - point.frame)
        vy_after = (after.y - point.y) / max(1, after.frame - point.frame)
        speed_before = math.hypot(vx_before, vy_before)
        speed_after = math.hypot(vx_after, vy_after)
        if speed_before < 2.0 or speed_after < 2.0:
            continue

        angle_change = vector_angle_change((vx_before, vy_before), (vx_after, vy_after))
        vertical_reversal = max(0.0, (vy_before - vy_after) / max(speed_before + speed_after, 1e-6))
        speed_ratio = min(speed_before, speed_after) / max(speed_before, speed_after)
        y_score = min(1.0, max(0.0, (point.y - min_y) / max(1.0, video_height - min_y)))
        score = (angle_change / 180.0) * 0.55 + vertical_reversal * 0.30 + speed_ratio * 0.10 + y_score * 0.05

        if angle_change >= min_angle_change and score >= min_score:
            candidates.append(
                ReceiveMoment(
                    frame=frame,
                    x=point.x,
                    y=point.y,
                    score=score,
                    angle_change_deg=angle_change,
                    speed_before=speed_before,
                    speed_after=speed_after,
                    vy_before=vy_before,
                    vy_after=vy_after,
                )
            )

    grouped = group_close_candidates(candidates, min_gap_frames=12)
    return sorted(grouped, key=lambda moment: moment.score, reverse=True)[:max_moments]


def smooth_points(points: list[BallPoint]) -> list[BallPoint]:
    by_frame = {point.frame: point for point in points}
    smoothed = []
    for point in points:
        neighbors = [
            by_frame[frame]
            for frame in range(point.frame - 2, point.frame + 3)
            if frame in by_frame
        ]
        if len(neighbors) < 3:
            smoothed.append(point)
            continue
        xs = sorted(item.x for item in neighbors)
        ys = sorted(item.y for item in neighbors)
        mid = len(neighbors) // 2
        smoothed.append(
            BallPoint(
                frame=point.frame,
                x=xs[mid],
                y=ys[mid],
                radius=point.radius,
                confidence=point.confidence,
            )
        )
    return smoothed


def nearest_point(
    by_frame: dict[int, BallPoint],
    target_frame: int,
    direction: int,
    max_gap: int,
) -> BallPoint | None:
    for offset in range(max_gap + 1):
        frame = target_frame + offset * direction
        point = by_frame.get(frame)
        if point is not None:
            return point
    return None


def vector_angle_change(a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    denom = math.hypot(ax, ay) * math.hypot(bx, by)
    if denom <= 1e-9:
        return 0.0
    cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / denom))
    return math.degrees(math.acos(cosine))


def group_close_candidates(candidates: list[ReceiveMoment], min_gap_frames: int) -> list[ReceiveMoment]:
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda moment: moment.frame)
    groups = [[ordered[0]]]
    for candidate in ordered[1:]:
        if candidate.frame - groups[-1][-1].frame <= min_gap_frames:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    return [max(group, key=lambda moment: moment.score) for group in groups]


def draw_frame(
    cv2: Any,
    frame: Any,
    frame_idx: int,
    fps: float,
    current_ball: BallPoint | None,
    ball_history: list[BallPoint],
    trail_length: int,
    moments: list[ReceiveMoment],
    hold_frames: int,
) -> Any:
    out = frame.copy()
    recent = [point for point in ball_history[-trail_length:] if frame_idx - point.frame <= trail_length]
    for idx in range(1, len(recent)):
        prev = recent[idx - 1]
        cur = recent[idx]
        age = max(0, frame_idx - cur.frame)
        thickness = max(2, int(8 * (1 - age / max(1, trail_length))))
        cv2.line(out, (int(prev.x), int(prev.y)), (int(cur.x), int(cur.y)), (255, 0, 255), thickness)
    for point in recent:
        age = max(0, frame_idx - point.frame)
        radius = max(3, int(8 * (1 - age / max(1, trail_length))))
        cv2.circle(out, (int(point.x), int(point.y)), radius, (255, 0, 255), -1)

    if current_ball:
        cv2.circle(out, (int(current_ball.x), int(current_ball.y)), int(max(10, current_ball.radius + 4)), (0, 255, 255), 4)

    active = [moment for moment in moments if abs(frame_idx - moment.frame) <= hold_frames]
    for moment in moments:
        color = (0, 0, 255) if moment in active else (80, 80, 180)
        cv2.drawMarker(
            out,
            (int(moment.x), int(moment.y)),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=36,
            thickness=3,
        )

    if active:
        best = min(active, key=lambda moment: abs(frame_idx - moment.frame))
        text = f"RECEIVE? {best.frame / fps:.2f}s score={best.score:.2f}"
        draw_label(cv2, out, text, 20, 42, (0, 0, 255), scale=1.0)
        cv2.circle(out, (int(best.x), int(best.y)), 34, (0, 0, 255), 4)
    else:
        draw_label(cv2, out, f"frame={frame_idx}", 20, 42, (30, 30, 30), scale=0.8)

    return out


def draw_label(
    cv2: Any,
    frame: Any,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float = 0.75,
) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 4)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def moment_to_dict(moment: ReceiveMoment, fps: float) -> dict[str, Any]:
    return {
        "frame": moment.frame,
        "time_sec": moment.frame / fps,
        "x": moment.x,
        "y": moment.y,
        "score": moment.score,
        "angle_change_deg": moment.angle_change_deg,
        "speed_before_px_per_frame": moment.speed_before,
        "speed_after_px_per_frame": moment.speed_after,
        "vy_before_px_per_frame": moment.vy_before,
        "vy_after_px_per_frame": moment.vy_after,
    }


if __name__ == "__main__":
    raise SystemExit(main())
