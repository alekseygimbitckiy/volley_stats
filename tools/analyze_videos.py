#!/usr/bin/env python3
"""Analyze volleyball clips with ball tracks, near-side player detections, and player embeddings.

This script intentionally keeps the model boundary explicit:

- ball positions are read from vball-net output files
- player detections are produced by Ultralytics YOLO
- player labels are assigned by matching crop descriptors to saved player embeddings
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Detection:
    frame: int
    bbox: tuple[float, float, float, float]
    confidence: float
    embedding: list[float]
    player_id: str | None = None
    player_distance: float | None = None


@dataclass
class Track:
    track_id: int
    detections: list[Detection] = field(default_factory=list)
    missed: int = 0

    @property
    def last_bbox(self) -> tuple[float, float, float, float]:
        return self.detections[-1].bbox

    @property
    def last_frame(self) -> int:
        return self.detections[-1].frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect near-side players and assign stored player labels.")
    parser.add_argument("--videos-dir", default="data/game")
    parser.add_argument("--layout", default="data/processed/calibrations/field_layout.json")
    parser.add_argument("--embeddings", default="data/processed/player_embeddings/player_embeddings.json")
    parser.add_argument("--ball-tracks-dir", default="data/processed/vball_net_raw")
    parser.add_argument("--output-dir", default="data/processed/analysis")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--match-threshold", type=float, default=0.65)
    parser.add_argument(
        "--vball-net-command",
        default=None,
        help=(
            "Optional command template to run ball tracking per video before analysis. "
            "Use {video}, {stem}, and {ball_tracks_dir} placeholders."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        deps = import_dependencies()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    videos = sorted(Path(args.videos_dir).glob("*.mp4"))
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print(f"No .mp4 files found in {args.videos_dir}", file=sys.stderr)
        return 1

    layout = load_layout(Path(args.layout))
    player_refs = load_player_embeddings(Path(args.embeddings))
    output_dir = Path(args.output_dir)
    per_video_dir = output_dir / "per_video"
    per_video_dir.mkdir(parents=True, exist_ok=True)

    print(f"Videos: {len(videos)}")
    print(f"Near-side polygon points: {len(layout)}")
    print(f"Player embeddings: {len(player_refs)}")

    yolo = deps["YOLO"](args.yolo_model)
    combined_rows: list[dict[str, Any]] = []

    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video_path.name}")
        if args.vball_net_command:
            run_vball_net(args.vball_net_command, video_path, Path(args.ball_tracks_dir))

        ball_track = load_ball_track(Path(args.ball_tracks_dir), video_path.stem)
        tracks = detect_and_track_players(
            deps=deps,
            model=yolo,
            video_path=video_path,
            near_polygon=layout,
            player_refs=player_refs,
            frame_stride=args.frame_stride,
            match_threshold=args.match_threshold,
        )

        video_result = {
            "video_id": video_path.stem,
            "video_path": str(video_path),
            "ball_positions": ball_track,
            "player_tracks": serialize_tracks(tracks),
        }
        result_path = per_video_dir / f"{video_path.stem}.json"
        result_path.write_text(json.dumps(video_result, indent=2) + "\n", encoding="utf-8")

        combined_rows.extend(flatten_rows(video_path.stem, ball_track, tracks))
        print(f"  ball positions: {len(ball_track)}")
        print(f"  player tracks: {len(tracks)}")
        print(f"  saved: {result_path}")

    csv_path = output_dir / "detections_summary.csv"
    write_summary_csv(csv_path, combined_rows)
    print(f"Summary: {csv_path}")
    return 0


def import_dependencies() -> dict[str, Any]:
    missing = []
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        cv2 = None
        missing.append("opencv-python")
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError:
        np = None
        missing.append("numpy")
    try:
        from ultralytics import YOLO  # type: ignore
    except ModuleNotFoundError:
        YOLO = None
        missing.append("ultralytics")

    if missing:
        raise RuntimeError(
            "Missing Python packages for video analysis: "
            + ", ".join(missing)
            + "\nInstall them in the venv, for example:\n"
            + "  python3 -m pip install opencv-python numpy ultralytics\n"
            + "Ball tracking also requires vball-net outputs in data/processed/vball_net_raw "
            + "or a --vball-net-command template."
        )
    return {"cv2": cv2, "np": np, "YOLO": YOLO}


def load_layout(path: Path) -> list[tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    points = data.get("near_side_boundary_px")
    if not points or len(points) < 3:
        raise SystemExit(f"Missing near_side_boundary_px in {path}")
    return [(float(point["x"]), float(point["y"])) for point in points]


def load_player_embeddings(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    refs = []
    for player in data.get("players", []):
        embedding = player.get("embedding") or []
        if embedding:
            refs.append({"player_id": player["player_id"], "embedding": [float(v) for v in embedding]})
    if not refs:
        raise SystemExit(f"No player embeddings found in {path}")
    return refs


def run_vball_net(command_template: str, video_path: Path, ball_tracks_dir: Path) -> None:
    ball_tracks_dir.mkdir(parents=True, exist_ok=True)
    if command_template.endswith(".onnx") or "{video}" not in command_template:
        command = [
            sys.executable,
            str(ROOT / "tools" / "run_vball_net.py"),
            str(video_path),
            "--model-path",
            command_template,
            "--output-dir",
            str(ball_tracks_dir),
        ]
        print(f"  running vball-net wrapper: {' '.join(command)}")
        completed = subprocess.run(command, text=True)
    else:
        command = command_template.format(
            video=str(video_path),
            stem=video_path.stem,
            ball_tracks_dir=str(ball_tracks_dir),
        )
        print(f"  running ball tracker: {command}")
        completed = subprocess.run(command, shell=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"vball-net command failed for {video_path.name}")


def load_ball_track(ball_tracks_dir: Path, stem: str) -> list[dict[str, Any]]:
    candidates = [
        ball_tracks_dir / f"{stem}.json",
        ball_tracks_dir / f"{stem}.csv",
        ball_tracks_dir / stem / "ball_positions.json",
        ball_tracks_dir / stem / "ball_positions.csv",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix == ".json":
                return normalize_ball_json(path)
            return normalize_ball_csv(path)
    print(f"  warning: no ball track file found for {stem}")
    return []


def normalize_ball_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("ball_positions", data if isinstance(data, list) else [])
    return [normalize_ball_row(row) for row in rows]


def normalize_ball_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [normalize_ball_row(row) for row in csv.DictReader(handle)]


def normalize_ball_row(row: dict[str, Any]) -> dict[str, Any]:
    frame = row.get("frame", row.get("frame_idx", row.get("frame_index", 0)))
    x = row.get("x", row.get("ball_x", row.get("ball_x_px", row.get("cx"))))
    y = row.get("y", row.get("ball_y", row.get("ball_y_px", row.get("cy"))))
    conf = row.get("confidence", row.get("conf", row.get("score", 1.0)))
    return {
        "frame": int(float(frame)),
        "x": None if x is None or x == "" else float(x),
        "y": None if y is None or y == "" else float(y),
        "confidence": float(conf or 0.0),
    }


def detect_and_track_players(
    deps: dict[str, Any],
    model: Any,
    video_path: Path,
    near_polygon: list[tuple[float, float]],
    player_refs: list[dict[str, Any]],
    frame_stride: int,
    match_threshold: float,
) -> list[Track]:
    cv2 = deps["cv2"]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  warning: could not open {video_path}")
        return []

    tracks: list[Track] = []
    next_track_id = 1
    frame_idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % frame_stride != 0:
            continue

        result = model.predict(frame, verbose=False, classes=[0], conf=0.25)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            foot = ((x1 + x2) / 2, y2)
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            if not (point_in_polygon(foot, near_polygon) or point_in_polygon(center, near_polygon)):
                continue
            crop = frame[max(0, int(y1)) : max(0, int(y2)), max(0, int(x1)) : max(0, int(x2))]
            if crop.size == 0:
                continue
            embedding = compute_crop_embedding(deps, crop)
            player_id, distance = match_player(embedding, player_refs, match_threshold)
            detections.append(
                Detection(
                    frame=frame_idx,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf[0]),
                    embedding=embedding,
                    player_id=player_id,
                    player_distance=distance,
                )
            )

        next_track_id = update_tracks(tracks, detections, next_track_id)

    cap.release()
    finalize_track_labels(tracks)
    return tracks


def update_tracks(tracks: list[Track], detections: list[Detection], next_track_id: int) -> int:
    unmatched = set(range(len(detections)))
    for track in tracks:
        best_idx = None
        best_iou = 0.0
        for idx in list(unmatched):
            score = iou(track.last_bbox, detections[idx].bbox)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_idx is not None and best_iou >= 0.25:
            track.detections.append(detections[best_idx])
            track.missed = 0
            unmatched.remove(best_idx)
        else:
            track.missed += 1

    for idx in unmatched:
        tracks.append(Track(track_id=next_track_id, detections=[detections[idx]]))
        next_track_id += 1

    tracks[:] = [track for track in tracks if track.missed <= 8 or len(track.detections) >= 2]
    return next_track_id


def finalize_track_labels(tracks: list[Track]) -> None:
    for track in tracks:
        votes: dict[str, int] = {}
        distances: dict[str, list[float]] = {}
        for det in track.detections:
            if not det.player_id:
                continue
            votes[det.player_id] = votes.get(det.player_id, 0) + 1
            distances.setdefault(det.player_id, []).append(float(det.player_distance or 1.0))
        if not votes:
            continue
        best = max(votes, key=votes.get)
        avg_distance = sum(distances[best]) / len(distances[best])
        for det in track.detections:
            det.player_id = best
            det.player_distance = avg_distance


def compute_crop_embedding(deps: dict[str, Any], crop: Any) -> list[float]:
    cv2 = deps["cv2"]
    np = deps["np"]
    resized = cv2.resize(crop, (16, 16), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32")

    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(rgb[:, :, channel], bins=8, range=(0, 256))
        hist_parts.extend((hist / 256.0).tolist())

    luma = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0
    pooled = cv2.resize(luma, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1).tolist()
    aspect = [min(3.0, crop.shape[1] / max(1, crop.shape[0])) / 3.0]
    return l2_normalize([float(v) for v in [*hist_parts, *pooled, *aspect]])


def match_player(
    embedding: list[float],
    player_refs: list[dict[str, Any]],
    threshold: float,
) -> tuple[str | None, float | None]:
    best_id = None
    best_distance = float("inf")
    for ref in player_refs:
        distance = euclidean(embedding, ref["embedding"])
        if distance < best_distance:
            best_id = ref["player_id"]
            best_distance = distance
    if best_distance > threshold:
        return None, best_distance
    return best_id, best_distance


def serialize_tracks(tracks: list[Track]) -> list[dict[str, Any]]:
    serialized = []
    for track in tracks:
        serialized.append(
            {
                "track_id": track.track_id,
                "detections": [
                    {
                        "frame": det.frame,
                        "bbox": {
                            "x1": det.bbox[0],
                            "y1": det.bbox[1],
                            "x2": det.bbox[2],
                            "y2": det.bbox[3],
                        },
                        "confidence": det.confidence,
                        "player_id": det.player_id,
                        "player_distance": det.player_distance,
                    }
                    for det in track.detections
                ],
            }
        )
    return serialized


def flatten_rows(video_id: str, ball_track: list[dict[str, Any]], tracks: list[Track]) -> list[dict[str, Any]]:
    rows = []
    for ball in ball_track:
        rows.append(
            {
                "video_id": video_id,
                "record_type": "ball",
                "frame": ball["frame"],
                "track_id": "",
                "player_id": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "ball_x": ball["x"],
                "ball_y": ball["y"],
                "confidence": ball["confidence"],
                "player_distance": "",
            }
        )
    for track in tracks:
        for det in track.detections:
            rows.append(
                {
                    "video_id": video_id,
                    "record_type": "player",
                    "frame": det.frame,
                    "track_id": track.track_id,
                    "player_id": det.player_id or "",
                    "x1": round(det.bbox[0], 3),
                    "y1": round(det.bbox[1], 3),
                    "x2": round(det.bbox[2], 3),
                    "y2": round(det.bbox[3], 3),
                    "ball_x": "",
                    "ball_y": "",
                    "confidence": round(det.confidence, 6),
                    "player_distance": "" if det.player_distance is None else round(det.player_distance, 6),
                }
            )
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "record_type",
        "frame",
        "track_id",
        "player_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "ball_x",
        "ball_y",
        "confidence",
        "player_distance",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, pi in enumerate(polygon):
        xi, yi = pi
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def euclidean(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return float("inf")
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [round(value / norm, 8) for value in values]


if __name__ == "__main__":
    raise SystemExit(main())
