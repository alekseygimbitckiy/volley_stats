#!/usr/bin/env python3
"""Single-video debug tracker for ball, near-side players, and saved player identities."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reid_osnet import OSNetEmbedder


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PlayerDetection:
    frame: int
    bbox: tuple[float, float, float, float]
    confidence: float
    embedding: list[float]
    player_id: str | None
    player_distance: float | None


@dataclass
class PlayerTrack:
    track_id: int
    detections: list[PlayerDetection] = field(default_factory=list)
    missed: int = 0

    @property
    def last_bbox(self) -> tuple[float, float, float, float]:
        return self.detections[-1].bbox


@dataclass
class BallObservation:
    frame: int
    x: float
    y: float
    radius: float
    confidence: float
    source: str


@dataclass
class BallState:
    last: BallObservation | None = None
    prev: BallObservation | None = None
    vx: float = 0.0
    vy: float = 0.0
    missed: int = 0

    def predict(self, frame: int) -> BallObservation | None:
        if self.last is None:
            return None
        dt = max(1, frame - self.last.frame)
        return BallObservation(
            frame=frame,
            x=self.last.x + self.vx * dt,
            y=self.last.y + self.vy * dt,
            radius=self.last.radius,
            confidence=max(0.05, self.last.confidence * (0.72 ** dt)),
            source="predicted",
        )

    def update(self, observation: BallObservation) -> None:
        if self.last is not None:
            dt = max(1, observation.frame - self.last.frame)
            measured_vx = (observation.x - self.last.x) / dt
            measured_vy = (observation.y - self.last.y) / dt
            alpha = 0.45
            self.vx = alpha * measured_vx + (1 - alpha) * self.vx
            self.vy = alpha * measured_vy + (1 - alpha) * self.vy
        self.prev = self.last
        self.last = observation
        self.missed = 0

    def mark_missed(self) -> None:
        self.missed += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug one video: ball, near team players, player labels.")
    parser.add_argument("video", help="Path to one MP4 clip.")
    parser.add_argument("--layout", default="data/processed/calibrations/field_layout.json")
    parser.add_argument("--embeddings", default="data/processed/player_embeddings/player_embeddings.json")
    parser.add_argument("--output-dir", default="data/processed/test_tracking")
    parser.add_argument("--ball-track", default=None, help="Existing normalized vball-net CSV/JSON for this video.")
    parser.add_argument("--vball-model-path", default=None, help="Run vball-net first with this ONNX model.")
    parser.add_argument("--vball-output-dir", default="data/processed/vball_net_raw")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--device", default="cpu", help="Ultralytics device, for example cpu, 0, or cuda:0.")
    parser.add_argument("--embedding-device", default="cpu", help="Torchreid device, for example cpu, 0, or cuda:0.")
    parser.add_argument("--frame-stride", type=int, default=3, help="Run player detector every N frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process the whole clip.")
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=1.05,
        help="Max L2 distance for assigning a player label. Use 0 to force nearest labels.",
    )
    parser.add_argument("--trail-length", type=int, default=45, help="Number of recent ball points to draw.")
    parser.add_argument("--max-ball-gap", type=int, default=18, help="Predict ball position through this many hidden frames.")
    parser.add_argument(
        "--team-filter",
        choices=["auto", "polygon", "largest", "lower-half", "none"],
        default="auto",
        help="How to keep only the closer team.",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip annotated MP4 export.")
    args = parser.parse_args()

    deps = import_dependencies()
    cv2 = deps["cv2"]

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = video_path.stem

    layout_data = load_layout_data(Path(args.layout))
    layout = extract_layout_polygon(layout_data)
    player_refs, embedding_info = load_player_embeddings(Path(args.embeddings))
    configure_embedding_backend(deps, embedding_info, args.embedding_device)
    external_ball_track = load_or_create_ball_track(args, video_path)
    model = deps["YOLO"](args.yolo_model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = args.max_frames if args.max_frames > 0 else frame_count

    writer = None
    annotated_path = output_dir / f"{output_stem}_annotated.mp4"
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))

    bg = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=18, detectShadows=False)
    player_tracks: list[PlayerTrack] = []
    ball_track: list[BallObservation] = []
    last_player_detections: list[PlayerDetection] = []
    ball_state = BallState()
    next_track_id = 1
    frame_idx = -1

    print(f"Video: {video_path}")
    print(f"Frames: {frame_count}, FPS: {fps:.2f}, size: {width}x{height}")
    print(f"Processing frames: {max_frames}")
    print(f"Team filter: {args.team_filter}")
    warn_layout_size_mismatch(layout_data, width, height)

    while frame_idx + 1 < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        player_boxes_for_ball: list[tuple[float, float, float, float]] = [
            det.bbox for det in last_player_detections
        ]

        yolo_ball_candidates: list[BallObservation] = []
        should_detect_players = frame_idx % max(1, args.frame_stride) == 0
        if should_detect_players:
            result = model.predict(frame, verbose=False, conf=0.20, device=args.device)[0]
            person_candidates = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                if cls_id == 0:
                    person_candidates.append((x1, y1, x2, y2, conf))
                elif cls_id == 32:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    radius = max(3.0, (x2 - x1 + y2 - y1) / 4)
                    yolo_ball_candidates.append(BallObservation(frame_idx, cx, cy, radius, conf, "yolo_sports_ball"))

            team_candidates = filter_team_candidates(
                person_candidates,
                layout,
                width,
                height,
                mode=args.team_filter,
            )
            detections = build_player_detections(
                deps=deps,
                frame=frame,
                frame_idx=frame_idx,
                candidates=team_candidates,
                player_refs=player_refs,
                threshold=args.match_threshold,
            )
            next_track_id = update_player_tracks(player_tracks, detections, next_track_id)
            if deps.get("reid_embedder") is None:
                finalize_track_labels(player_tracks)
            last_player_detections = detections
            player_boxes_for_ball = [det.bbox for det in last_player_detections]

        external_ball = external_ball_track.get(frame_idx)
        if external_ball is not None:
            ball_state.update(external_ball)
            ball = external_ball
        else:
            motion_candidates = detect_motion_ball_candidates(
                deps=deps,
                frame=frame,
                bg=bg,
                near_polygon=layout,
                player_boxes=player_boxes_for_ball,
                frame_idx=frame_idx,
                predicted_ball=ball_state.predict(frame_idx),
            )
            ball = update_ball_state(
                candidates=[*yolo_ball_candidates, *motion_candidates],
                state=ball_state,
                frame_idx=frame_idx,
                max_gap=args.max_ball_gap,
            )
        if ball:
            ball_track.append(ball)

        if writer:
            annotated = draw_debug_frame(
                deps=deps,
                frame=frame,
                near_polygon=layout,
                tracks=player_tracks,
                current_frame=frame_idx,
                ball=ball,
                ball_track=ball_track,
                team_filter=args.team_filter,
                trail_length=args.trail_length,
            )
            writer.write(annotated)

        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}: tracks={len(player_tracks)} ball_points={len(ball_track)}")

    cap.release()
    if writer:
        writer.release()

    result = {
        "video_id": output_stem,
        "video_path": str(video_path),
        "frame_count_processed": frame_idx + 1,
        "team_filter": args.team_filter,
        "ball_track": [ball.__dict__ for ball in ball_track],
        "player_tracks": serialize_tracks(player_tracks),
    }
    json_path = output_dir / f"{output_stem}_test_tracking.json"
    csv_path = output_dir / f"{output_stem}_test_tracking.csv"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, output_stem, ball_track, player_tracks)

    print(f"Ball positions: {len(ball_track)}")
    print(f"Player tracks: {len(player_tracks)}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    if writer:
        print(f"Saved annotated video: {annotated_path}")

    if not ball_track:
        print(
            "Warning: no ball positions found. Try --vball-model-path with a vball-net ONNX model, "
            "or use --team-filter none for fallback debugging."
        )
    if not player_tracks:
        print("Warning: no player tracks found. Your saved field polygon may be too narrow; try --team-filter largest.")
    return 0


def load_or_create_ball_track(
    args: argparse.Namespace,
    video_path: Path,
) -> dict[int, BallObservation]:
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
        print("Running vball-net before debug tracking:")
        print(" ".join(command))
        completed = subprocess.run(command, text=True)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        args.ball_track = str(Path(args.vball_output_dir) / f"{video_path.stem}.csv")

    if args.ball_track:
        path = Path(args.ball_track)
    else:
        candidate_csv = Path(args.vball_output_dir) / f"{video_path.stem}.csv"
        candidate_json = Path(args.vball_output_dir) / f"{video_path.stem}.json"
        if candidate_csv.exists():
            path = candidate_csv
        elif candidate_json.exists():
            path = candidate_json
        else:
            return {}

    if not path.exists():
        raise SystemExit(f"Ball track file not found: {path}")
    rows = load_ball_track_file(path)
    print(f"Loaded external ball track: {path} ({len(rows)} visible frames)")
    return rows


def load_ball_track_file(path: Path) -> dict[int, BallObservation]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("ball_positions", data if isinstance(data, list) else [])
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    track = {}
    for row in rows:
        frame = int(float(row.get("frame", row.get("Frame", 0)) or 0))
        confidence = float(row.get("confidence", row.get("Confidence", 0)) or 0)
        x = row.get("x", row.get("X"))
        y = row.get("y", row.get("Y"))
        if x in (None, "", "-1") or y in (None, "", "-1"):
            continue
        track[frame] = BallObservation(
            frame=frame,
            x=float(x),
            y=float(y),
            radius=max(5.0, float(row.get("width", row.get("W", 10)) or 10) / 2),
            confidence=confidence,
            source=str(row.get("source", "external_ball_track") or "external_ball_track"),
        )
    return track


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
        raise SystemExit(
            "Missing packages: "
            + ", ".join(missing)
            + "\nInstall with: python3 -m pip install opencv-python numpy ultralytics"
        )
    return {"cv2": cv2, "np": np, "YOLO": YOLO}


def load_layout_data(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def extract_layout_polygon(data: dict) -> list[tuple[float, float]]:
    points = data.get("near_side_boundary_px")
    if not points or len(points) < 3:
        raise SystemExit("Missing near_side_boundary_px in layout")
    return [(float(point["x"]), float(point["y"])) for point in points]


def warn_layout_size_mismatch(layout_data: dict, video_width: int, video_height: int) -> None:
    frame = layout_data.get("frame") or {}
    layout_width = int(frame.get("width_px") or 0)
    layout_height = int(frame.get("height_px") or 0)
    if not layout_width or not layout_height:
        return
    if layout_width != video_width or layout_height != video_height:
        print(
            "Warning: layout frame size does not match video size. "
            f"layout={layout_width}x{layout_height}, video={video_width}x{video_height}. "
            "Court filtering may remove correct ball/player detections; remark the court for this camera orientation."
        )


def load_player_embeddings(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    refs = []
    for player in data.get("players", []):
        embedding = player.get("embedding") or []
        if embedding:
            refs.append({"player_id": player["player_id"], "embedding": [float(v) for v in embedding]})
    if not refs:
        raise SystemExit(f"No player embeddings found in {path}")
    return refs, data.get("embedding", {})


def configure_embedding_backend(deps: dict[str, Any], embedding_info: dict[str, Any], device: str) -> None:
    embedding_type = str(embedding_info.get("type", ""))
    deps["embedding_type"] = embedding_type
    if embedding_type.startswith("torchreid_osnet"):
        print(f"Loading player ReID embedder: {embedding_type} on {device}")
        deps["reid_embedder"] = OSNetEmbedder(device=device)
    else:
        deps["reid_embedder"] = None


def filter_team_candidates(
    candidates: list[tuple[float, float, float, float, float]],
    polygon: list[tuple[float, float]],
    width: int,
    height: int,
    mode: str,
) -> list[tuple[float, float, float, float, float]]:
    if mode == "none":
        return candidates
    if mode == "polygon":
        return [candidate for candidate in candidates if candidate_in_polygon(candidate, polygon)]
    if mode == "lower-half":
        return [candidate for candidate in candidates if candidate[3] >= height * 0.48]
    if mode == "largest":
        return largest_candidates(candidates)

    polygon_hits = [candidate for candidate in candidates if candidate_in_polygon(candidate, polygon)]
    if polygon_hits:
        return polygon_hits
    lower_hits = [candidate for candidate in candidates if candidate[3] >= height * 0.48]
    return largest_candidates(lower_hits or candidates)


def largest_candidates(
    candidates: list[tuple[float, float, float, float, float]],
    limit: int = 6,
) -> list[tuple[float, float, float, float, float]]:
    return sorted(candidates, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)[:limit]


def candidate_in_polygon(
    candidate: tuple[float, float, float, float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    x1, y1, x2, y2, _ = candidate
    foot = ((x1 + x2) / 2, y2)
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    return point_in_polygon(foot, polygon) or point_in_polygon(center, polygon)


def build_player_detections(
    deps: dict[str, Any],
    frame: Any,
    frame_idx: int,
    candidates: list[tuple[float, float, float, float, float]],
    player_refs: list[dict[str, Any]],
    threshold: float,
) -> list[PlayerDetection]:
    detections = []
    for x1, y1, x2, y2, confidence in candidates:
        crop = frame[max(0, int(y1)) : max(0, int(y2)), max(0, int(x1)) : max(0, int(x2))]
        if crop.size == 0:
            continue
        embedding = compute_crop_embedding(deps, crop)
        detections.append(
            PlayerDetection(
                frame=frame_idx,
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                embedding=embedding,
                player_id=None,
                player_distance=None,
            )
        )
    assign_nearest_players_to_detections(detections, player_refs, threshold)
    return detections


def assign_nearest_players_to_detections(
    detections: list[PlayerDetection],
    player_refs: list[dict[str, Any]],
    threshold: float,
) -> None:
    for detection in detections:
        detection.player_id = None
        detection.player_distance = None

    pairs = []
    for ref in player_refs:
        for detection in detections:
            pairs.append((euclidean(detection.embedding, ref["embedding"]), ref["player_id"], detection))
    pairs.sort(key=lambda item: item[0])

    used_players = set()
    used_detections = set()
    for distance, player_id, detection in pairs:
        if threshold > 0 and distance > threshold:
            continue
        if player_id in used_players or id(detection) in used_detections:
            continue
        detection.player_id = player_id
        detection.player_distance = distance
        used_players.add(player_id)
        used_detections.add(id(detection))


def keep_nearest_detection_per_player(detections: list[PlayerDetection]) -> None:
    best_by_player: dict[str, PlayerDetection] = {}
    for detection in detections:
        if detection.player_id is None or detection.player_distance is None:
            continue
        current_best = best_by_player.get(detection.player_id)
        if current_best is None or detection.player_distance < (current_best.player_distance or float("inf")):
            best_by_player[detection.player_id] = detection

    for detection in detections:
        if detection.player_id is None:
            continue
        if best_by_player.get(detection.player_id) is not detection:
            detection.player_id = None
            detection.player_distance = None


def detect_motion_ball_candidates(
    deps: dict[str, Any],
    frame: Any,
    bg: Any,
    near_polygon: list[tuple[float, float]],
    player_boxes: list[tuple[float, float, float, float]],
    frame_idx: int,
    predicted_ball: BallObservation | None,
) -> list[BallObservation]:
    cv2 = deps["cv2"]
    np = deps["np"]
    fg = bg.apply(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([180, 85, 255]))
    yellow = cv2.inRange(hsv, np.array([18, 45, 110]), np.array([45, 255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_and(fg, white), cv2.bitwise_and(fg, yellow))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 12 or area > 650:
            continue
        x_box, y_box, w_box, h_box = cv2.boundingRect(contour)
        aspect = w_box / max(1, h_box)
        if aspect < 0.35 or aspect > 2.8:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < 2 or radius > 18:
            continue
        if near_polygon and not point_in_polygon((x, y), near_polygon):
            if predicted_ball is None:
                continue
        if any(point_inside_bbox((x, y), expand_bbox(box, 0.05)) for box in player_boxes):
            continue
        circularity = area / (math.pi * radius * radius + 1e-9)
        if circularity < 0.25:
            continue
        if predicted_ball is not None:
            jump = math.hypot(x - predicted_ball.x, y - predicted_ball.y)
            max_jump = 170 + 28 * max(1, frame_idx - predicted_ball.frame)
            if jump > max_jump:
                continue
        confidence = min(1.0, 0.25 + circularity * 0.5 + min(0.25, area / 4000))
        candidates.append(BallObservation(frame_idx, float(x), float(y), float(radius), confidence, "motion_color"))
    return candidates


def update_ball_state(
    candidates: list[BallObservation],
    state: BallState,
    frame_idx: int,
    max_gap: int,
) -> BallObservation | None:
    predicted = state.predict(frame_idx)
    accepted = choose_ball_candidate(candidates, predicted, has_track=state.last is not None)
    if accepted is not None:
        state.update(accepted)
        return accepted

    state.mark_missed()
    if predicted is not None and state.missed <= max_gap:
        state.last = predicted
        return predicted
    return None


def choose_ball_candidate(
    candidates: list[BallObservation],
    predicted: BallObservation | None,
    has_track: bool,
) -> BallObservation | None:
    if not candidates:
        return None
    if not has_track or predicted is None:
        best_initial = max(candidates, key=lambda candidate: candidate.confidence)
        return best_initial if best_initial.confidence >= 0.34 else None

    def score(candidate: BallObservation) -> float:
        distance = math.hypot(candidate.x - predicted.x, candidate.y - predicted.y)
        expected_limit = 130 + 22 * max(1, candidate.frame - predicted.frame)
        distance_penalty = min(1.0, distance / expected_limit)
        source_bonus = 0.12 if candidate.source == "yolo_sports_ball" else 0.0
        return candidate.confidence + source_bonus - 0.65 * distance_penalty

    best = max(candidates, key=score)
    if score(best) < 0.02:
        return None
    return best


def update_player_tracks(
    tracks: list[PlayerTrack],
    detections: list[PlayerDetection],
    next_track_id: int,
) -> int:
    unmatched = set(range(len(detections)))
    for track in tracks:
        best_idx = None
        best_iou = 0.0
        for idx in list(unmatched):
            score = iou(track.last_bbox, detections[idx].bbox)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_idx is not None and best_iou >= 0.22:
            track.detections.append(detections[best_idx])
            track.missed = 0
            unmatched.remove(best_idx)
        else:
            track.missed += 1
    for idx in unmatched:
        tracks.append(PlayerTrack(track_id=next_track_id, detections=[detections[idx]]))
        next_track_id += 1
    tracks[:] = [track for track in tracks if track.missed <= 12 or len(track.detections) >= 2]
    return next_track_id


def finalize_track_labels(tracks: list[PlayerTrack]) -> None:
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


def draw_debug_frame(
    deps: dict[str, Any],
    frame: Any,
    near_polygon: list[tuple[float, float]],
    tracks: list[PlayerTrack],
    current_frame: int,
    ball: BallObservation | None,
    ball_track: list[BallObservation],
    team_filter: str,
    trail_length: int,
) -> Any:
    cv2 = deps["cv2"]
    np = deps["np"]
    out = frame.copy()
    if near_polygon:
        pts = np.array(near_polygon, dtype=np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 220, 120), thickness=3)
    visible_detections = []
    for track in tracks:
        det = latest_detection_near_frame(track, current_frame, max_gap=12)
        if not det:
            continue
        visible_detections.append((track, det))

    unique_player_detection_ids = nearest_visible_detection_ids_by_player(visible_detections)
    for track, det in visible_detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        use_player_label = det.player_id is not None and id(det) in unique_player_detection_ids
        label = det.player_id if use_player_label else f"track_{track.track_id}"
        if use_player_label and det.player_distance is not None:
            label = f"{label} d={det.player_distance:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 160, 0), 3)
        cv2.putText(out, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(out, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 70, 180), 2)
    if ball:
        recent = [point for point in ball_track[-trail_length:] if current_frame - point.frame <= trail_length]
        for idx in range(1, len(recent)):
            prev = recent[idx - 1]
            cur = recent[idx]
            age = current_frame - cur.frame
            thickness = max(3, int(12 * (1 - age / max(1, trail_length))))
            cv2.line(out, (int(prev.x), int(prev.y)), (int(cur.x), int(cur.y)), (255, 0, 255), thickness)
        for point in recent:
            age = current_frame - point.frame
            radius = max(3, int(9 * (1 - age / max(1, trail_length))))
            cv2.circle(out, (int(point.x), int(point.y)), radius, (255, 0, 255), -1)
        cv2.circle(out, (int(ball.x), int(ball.y)), int(max(10, ball.radius + 4)), (0, 255, 255), 4)
        cv2.putText(
            out,
            f"ball {ball.source} {ball.confidence:.2f}",
            (int(ball.x) + 10, int(ball.y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            out,
            f"ball {ball.source} {ball.confidence:.2f}",
            (int(ball.x) + 10, int(ball.y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    cv2.putText(
        out,
        f"frame={current_frame} team_filter={team_filter}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        3,
    )
    cv2.putText(out, f"frame={current_frame} team_filter={team_filter}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return out


def nearest_visible_detection_ids_by_player(
    visible_detections: list[tuple[PlayerTrack, PlayerDetection]],
) -> set[int]:
    best_by_player: dict[str, PlayerDetection] = {}
    for _, detection in visible_detections:
        if detection.player_id is None or detection.player_distance is None:
            continue
        current_best = best_by_player.get(detection.player_id)
        if current_best is None or detection.player_distance < (current_best.player_distance or float("inf")):
            best_by_player[detection.player_id] = detection
    return {id(detection) for detection in best_by_player.values()}


def latest_detection_near_frame(
    track: PlayerTrack,
    frame: int,
    max_gap: int,
) -> PlayerDetection | None:
    if not track.detections:
        return None
    det = track.detections[-1]
    if abs(frame - det.frame) <= max_gap:
        return det
    return None


def compute_crop_embedding(deps: dict[str, Any], crop: Any) -> list[float]:
    if deps.get("reid_embedder") is not None:
        return deps["reid_embedder"].embed_bgr(deps["cv2"], crop)

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
    refs: list[dict[str, Any]],
    threshold: float,
) -> tuple[str | None, float | None]:
    best_id = None
    best_distance = float("inf")
    for ref in refs:
        distance = euclidean(embedding, ref["embedding"])
        if distance < best_distance:
            best_id = ref["player_id"]
            best_distance = distance
    if best_distance > threshold:
        return None, best_distance
    return best_id, best_distance


def serialize_tracks(tracks: list[PlayerTrack]) -> list[dict[str, Any]]:
    return [
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
        for track in tracks
    ]


def write_csv(
    path: Path,
    video_id: str,
    ball_track: list[BallObservation],
    tracks: list[PlayerTrack],
) -> None:
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
        "source",
        "player_distance",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ball in ball_track:
            writer.writerow(
                {
                    "video_id": video_id,
                    "record_type": "ball",
                    "frame": ball.frame,
                    "track_id": "",
                    "player_id": "",
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "ball_x": round(ball.x, 3),
                    "ball_y": round(ball.y, 3),
                    "confidence": round(ball.confidence, 6),
                    "source": ball.source,
                    "player_distance": "",
                }
            )
        for track in tracks:
            for det in track.detections:
                writer.writerow(
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
                        "source": "yolo_person",
                        "player_distance": "" if det.player_distance is None else round(det.player_distance, 6),
                    }
                )


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


def expand_polygon(points: list[tuple[float, float]], margin: float) -> list[tuple[float, float]]:
    if margin <= 0:
        return points
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    expanded = []
    for x, y in points:
        dx = x - cx
        dy = y - cy
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        expanded.append((x + margin * dx / length, y + margin * dy / length))
    return expanded


def point_inside_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def expand_bbox(
    bbox: tuple[float, float, float, float],
    ratio: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return x1 - w * ratio, y1 - h * ratio, x2 + w * ratio, y2 + h * ratio


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
