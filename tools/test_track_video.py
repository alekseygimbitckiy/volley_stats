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

from jersey_ocr import JerseyOCR, OCRIdentity, normalize_name, normalize_number as normalize_jersey_number
from player_roster import PlayerRoster, load_roster
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
    jersey_number: str | None = None
    jersey_confidence: float | None = None
    shirt_name: str | None = None
    shirt_name_confidence: float | None = None
    identity_source: str | None = None


@dataclass
class PlayerTrack:
    track_id: int | str
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


@dataclass
class UniformColorFilter:
    enabled: bool
    threshold: float
    warmup_frames: int
    min_samples: int
    max_samples: int
    samples: list[tuple[float, float, float]] = field(default_factory=list)
    profile: tuple[float, float, float] | None = None
    frames_seen: int = 0
    source: str = "warmup"

    def __post_init__(self) -> None:
        if self.samples:
            # The actual median is computed after dependencies are available.
            self.source = "embedding-snapshots"

    def filter_candidates(
        self,
        deps: dict[str, Any],
        frame: Any,
        candidates: list[tuple[float, float, float, float, float]],
    ) -> list[tuple[float, float, float, float, float]]:
        if not self.enabled or not candidates:
            return candidates

        colored = []
        for candidate in candidates:
            color = extract_uniform_color(deps, frame, candidate)
            if color is not None:
                colored.append((candidate, color))

        if not colored:
            return candidates

        self.frames_seen += 1
        if self.profile is None and self.samples:
            self.profile = median_hsv(self.samples, deps["np"])
            print(f"Uniform color profile HSV from {self.source}: {format_hsv(self.profile)}")

        if self.profile is None:
            self.samples.extend(color for _candidate, color in colored)
            self.samples = self.samples[-self.max_samples :]
            if len(self.samples) >= self.min_samples or self.frames_seen >= self.warmup_frames:
                self.profile = median_hsv(self.samples, deps["np"])
                self.source = "video-warmup"
                print(f"Uniform color profile HSV from {self.source}: {format_hsv(self.profile)}")
            return candidates

        accepted = [
            candidate
            for candidate, color in colored
            if hsv_distance(color, self.profile) <= self.threshold
        ]
        if accepted:
            self.samples.extend(color for candidate, color in colored if candidate in accepted)
            self.samples = self.samples[-self.max_samples :]
            self.profile = median_hsv(self.samples, deps["np"])
        return accepted


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
    parser.add_argument("--players", choices=["auto", "off"], default="auto", help="Use player detection/tracking. Set off for ball-only debug videos.")
    parser.add_argument("--reid", choices=["auto", "off"], default="auto", help="Use stored player embeddings for ReID fallback labeling.")
    parser.add_argument("--ocr", choices=["auto", "off"], default="auto", help="Prefer jersey-number OCR before ReID matching.")
    parser.add_argument("--ocr-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ocr-backend", choices=["easyocr", "paddleocr"], default="easyocr")
    parser.add_argument("--ocr-languages", default="en", help="Comma-separated EasyOCR languages, for example en or en,ru.")
    parser.add_argument("--ocr-min-confidence", type=float, default=0.35)
    parser.add_argument(
        "--ocr-relabel-min-confidence",
        type=float,
        default=0.85,
        help="Minimum OCR confidence required to correct an existing tracked player label.",
    )
    parser.add_argument(
        "--ocr-skip-overlap-iou",
        type=float,
        default=0.25,
        help="Do not run OCR for a player box that overlaps another player box by at least this IoU.",
    )
    parser.add_argument(
        "--ocr-relabel-max-center-jump",
        "--reid-relabel-max-center-jump",
        dest="reid_relabel_max_center_jump",
        type=float,
        default=160.0,
        help="Reject ReID roster-fill labels whose box center is too far from that player's smooth predicted center.",
    )
    parser.add_argument("--ocr-every-n-frames", type=int, default=15, help="Run jersey OCR every N frames; embeddings still run every detection frame.")
    parser.add_argument("--roster", default=None, help="JSON file with real player_id, jersey_number, and names/aliases.")
    parser.add_argument("--roster-name-threshold", type=float, default=0.78)
    parser.add_argument("--roster-name-margin", type=float, default=0.05)
    parser.add_argument("--tracker", choices=["bytetrack", "deepsort", "iou"], default="deepsort")
    parser.add_argument("--bytetrack-high-thresh", type=float, default=0.25)
    parser.add_argument("--bytetrack-low-thresh", type=float, default=0.10)
    parser.add_argument("--bytetrack-new-track-thresh", type=float, default=0.25)
    parser.add_argument("--bytetrack-match-thresh", type=float, default=0.80)
    parser.add_argument("--bytetrack-track-buffer", type=int, default=30)
    parser.add_argument("--deepsort-max-age", type=int, default=30)
    parser.add_argument("--deepsort-n-init", type=int, default=1)
    parser.add_argument("--deepsort-max-cosine-distance", type=float, default=0.7)
    parser.add_argument("--frame-stride", type=int, default=3, help="Run player detector every N frames.")
    parser.add_argument("--person-nms-iou", type=float, default=0.55, help="Suppress duplicate person boxes above this IoU.")
    parser.add_argument(
        "--uniform-color-filter",
        action="store_true",
        help="Filter opponent player boxes by near-team uniform color before OCR/ReID labeling.",
    )
    parser.add_argument("--uniform-color-threshold", type=float, default=48.0, help="Max HSV distance from the learned near-team uniform color.")
    parser.add_argument("--uniform-color-warmup-frames", type=int, default=30, help="Detection frames used to learn near-team color before filtering.")
    parser.add_argument("--uniform-color-min-samples", type=int, default=18, help="Minimum player crops needed before the uniform color filter activates.")
    parser.add_argument("--uniform-color-max-samples", type=int, default=120, help="Rolling sample count used to update near-team color.")
    parser.add_argument(
        "--uniform-color-source",
        choices=["embeddings", "warmup"],
        default="embeddings",
        help="Learn target uniform color from saved embedding snapshots, or from early video detections.",
    )
    parser.add_argument(
        "--fill-roster-labels",
        action="store_true",
        help="Force visible near-side tracks to use the roster labels when OCR/ReID leaves a track unlabeled.",
    )
    parser.add_argument(
        "--predict-missing-players",
        action="store_true",
        help="Predict missing roster-labeled player boxes from track velocity so labels stay visible through short detector misses.",
    )
    parser.add_argument(
        "--max-player-prediction-gap",
        type=int,
        default=45,
        help="Maximum frames to keep predicting a roster-labeled player after detector/tracker misses.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process the whole clip.")
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=1.05,
        help="Max L2 distance for assigning a player label. Use 0 to force nearest labels.",
    )
    parser.add_argument("--trail-length", type=int, default=45, help="Number of recent ball points to draw.")
    parser.add_argument("--max-ball-gap", type=int, default=18, help="Predict ball position through this many hidden frames.")
    parser.add_argument("--ball-max-jump", type=float, default=95.0, help="Reject ball detections farther than this many pixels from the predicted ball position in one frame.")
    parser.add_argument(
        "--ball-reacquire-gap",
        type=int,
        default=5,
        help="After this many consecutive missing ball frames, use --ball-reacquire-max-jump for the next accepted real detection.",
    )
    parser.add_argument(
        "--ball-reacquire-max-jump",
        type=float,
        default=1000.0,
        help="Temporary ball jump limit used after --ball-reacquire-gap missing frames.",
    )
    parser.add_argument(
        "--ball-reset-gap",
        type=int,
        default=45,
        help="After this many consecutive rejected/missing ball frames, reset ball state so tracking can restart anywhere.",
    )
    parser.add_argument(
        "--ball-source",
        choices=["auto", "vball-net", "yolo", "motion"],
        default="auto",
        help="Which ball detections to use before optional prediction.",
    )
    parser.add_argument(
        "--team-filter",
        choices=["auto", "polygon", "court-intersection", "court-nearest-6", "largest", "lower-half", "none"],
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
    players_enabled = args.players != "off"
    player_refs: list[dict[str, Any]] = []
    embedding_info: dict[str, Any] = {}
    roster = None
    if players_enabled:
        player_refs, embedding_info = load_player_embeddings(Path(args.embeddings))
        roster = load_roster(args.roster, args.roster_name_threshold, args.roster_name_margin)
        if roster is not None:
            player_refs = merge_roster_refs(player_refs, roster)
            print(f"Loaded roster players: {len(roster.players)}")
        configure_embedding_backend(deps, embedding_info, args.embedding_device, enabled=args.reid != "off")
        configure_ocr_backend(deps, args, embedding_info, player_refs)
    else:
        deps["reid_embedder"] = None
        print("Player detection disabled; producing ball-only output.")
    external_ball_track = load_or_create_ball_track(args, video_path)
    model = deps["YOLO"](args.yolo_model)
    player_tracker = create_player_tracker(args) if players_enabled else None
    uniform_samples = []
    if players_enabled and args.uniform_color_filter and args.uniform_color_source == "embeddings":
        uniform_samples = load_uniform_color_samples(Path(args.embeddings), deps, args.uniform_color_max_samples)
    uniform_filter = UniformColorFilter(
        enabled=args.uniform_color_filter,
        threshold=args.uniform_color_threshold,
        warmup_frames=args.uniform_color_warmup_frames,
        min_samples=args.uniform_color_min_samples,
        max_samples=args.uniform_color_max_samples,
        samples=uniform_samples,
    )

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
    visible_player_detections: list[dict[str, Any]] = []
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
        should_detect_model = (
            (players_enabled and (args.tracker == "bytetrack" or frame_idx % max(1, args.frame_stride) == 0))
            or args.ball_source in ("auto", "yolo")
        )
        detections: list[PlayerDetection] = []
        if should_detect_model:
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

            if not players_enabled:
                person_candidates = []
                team_candidates = []
            else:
                person_candidates = suppress_duplicate_person_boxes(person_candidates, args.person_nms_iou)
                team_candidates = filter_team_candidates(
                    person_candidates,
                    layout,
                    width,
                    height,
                    mode=args.team_filter,
                )
                team_candidates = uniform_filter.filter_candidates(deps, frame, team_candidates)
                detections = build_player_detections(
                    deps=deps,
                    frame=frame,
                    frame_idx=frame_idx,
                    candidates=team_candidates,
                    player_refs=player_refs,
                    threshold=args.match_threshold,
                    ocr_enabled=should_run_ocr(args, frame_idx),
                    ocr_overlap_iou=args.ocr_skip_overlap_iou,
                    roster=roster,
                    use_reid=args.reid != "off",
                )
        if players_enabled and args.tracker == "bytetrack":
            update_bytetrack_player_tracks(
                player_tracker,
                player_tracks,
                detections,
                frame_idx,
                frame,
                max_missed=max_tracker_missed(args),
                ocr_relabel_min_confidence=args.ocr_relabel_min_confidence,
            )
            fill_current_roster_labels(player_tracks, frame_idx, roster, player_refs, args.fill_roster_labels, use_reid=args.reid != "off", reid_max_center_jump=args.reid_relabel_max_center_jump)
            predict_missing_roster_tracks(player_tracks, frame_idx, roster, width, height, args.predict_missing_players)
            fill_current_roster_labels(player_tracks, frame_idx, roster, player_refs, args.fill_roster_labels, use_reid=args.reid != "off", reid_max_center_jump=args.reid_relabel_max_center_jump)
            last_player_detections = current_tracker_detections(player_tracks, frame_idx)
            player_boxes_for_ball = [det.bbox for det in last_player_detections]
        elif players_enabled and args.tracker == "deepsort":
            update_deepsort_player_tracks(
                player_tracker,
                player_tracks,
                detections,
                frame_idx,
                max_missed=max_tracker_missed(args),
                ocr_relabel_min_confidence=args.ocr_relabel_min_confidence,
            )
            fill_current_roster_labels(player_tracks, frame_idx, roster, player_refs, args.fill_roster_labels, use_reid=args.reid != "off", reid_max_center_jump=args.reid_relabel_max_center_jump)
            predict_missing_roster_tracks(player_tracks, frame_idx, roster, width, height, args.predict_missing_players)
            fill_current_roster_labels(player_tracks, frame_idx, roster, player_refs, args.fill_roster_labels, use_reid=args.reid != "off", reid_max_center_jump=args.reid_relabel_max_center_jump)
            last_player_detections = current_tracker_detections(player_tracks, frame_idx)
            player_boxes_for_ball = [det.bbox for det in last_player_detections]
        elif players_enabled and should_detect_model:
            next_track_id = update_player_tracks(
                player_tracks,
                detections,
                next_track_id,
                max_missed=max_tracker_missed(args),
            )
            if deps.get("reid_embedder") is None:
                finalize_track_labels(player_tracks)
            fill_current_roster_labels(player_tracks, frame_idx, roster, player_refs, args.fill_roster_labels, use_reid=args.reid != "off", reid_max_center_jump=args.reid_relabel_max_center_jump)
            predict_missing_roster_tracks(player_tracks, frame_idx, roster, width, height, args.predict_missing_players)
            fill_current_roster_labels(player_tracks, frame_idx, roster, player_refs, args.fill_roster_labels, use_reid=args.reid != "off", reid_max_center_jump=args.reid_relabel_max_center_jump)
            last_player_detections = current_tracker_detections(player_tracks, frame_idx)
            player_boxes_for_ball = [det.bbox for det in last_player_detections]

        external_ball = external_ball_track.get(frame_idx)
        ball_candidates: list[BallObservation] = []
        if args.ball_source in ("auto", "yolo"):
            ball_candidates.extend(yolo_ball_candidates)
        if args.ball_source in ("auto", "motion"):
            ball_candidates.extend(
                detect_motion_ball_candidates(
                    deps=deps,
                    frame=frame,
                    bg=bg,
                    near_polygon=layout,
                    player_boxes=player_boxes_for_ball,
                    frame_idx=frame_idx,
                    predicted_ball=ball_state.predict(frame_idx),
                )
            )
        if args.ball_source in ("auto", "vball-net") and external_ball is not None:
            ball_candidates.insert(0, external_ball)
        ball = update_ball_state(
            candidates=ball_candidates,
            state=ball_state,
            frame_idx=frame_idx,
            max_gap=args.max_ball_gap,
            max_jump=args.ball_max_jump,
            reacquire_gap=args.ball_reacquire_gap,
            reacquire_max_jump=args.ball_reacquire_max_jump,
            reset_gap=args.ball_reset_gap,
        )
        if ball:
            ball_track.append(ball)

        if players_enabled:
            visible_player_detections.append(
                serialize_visible_player_detections(player_tracks, frame_idx, max_gap=12)
            )

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
        "visible_player_detections": visible_player_detections,
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
        rows = data.get("ball_track", data.get("ball_positions", data if isinstance(data, list) else []))
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
        radius = row.get("radius")
        if radius in (None, ""):
            radius = max(5.0, float(row.get("width", row.get("W", 10)) or 10) / 2)
        track[frame] = BallObservation(
            frame=frame,
            x=float(x),
            y=float(y),
            radius=float(radius),
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
        jersey_number = normalize_jersey_number(player.get("jersey_number"))
        shirt_name = normalize_name(player.get("shirt_name") or "")
        if embedding or jersey_number or shirt_name:
            refs.append(
                {
                    "player_id": player["player_id"],
                    "embedding": [float(v) for v in embedding],
                    "jersey_number": jersey_number,
                    "jersey_number_confidence": player.get("jersey_number_confidence"),
                    "shirt_name": shirt_name,
                    "shirt_name_confidence": player.get("shirt_name_confidence"),
                }
            )
    if not refs:
        raise SystemExit(f"No player embeddings found in {path}")
    return refs, data.get("embedding", {})


def load_uniform_color_samples(
    embeddings_path: Path,
    deps: dict[str, Any],
    limit: int,
) -> list[tuple[float, float, float]]:
    cv2 = deps["cv2"]
    data = json.loads(embeddings_path.read_text(encoding="utf-8"))
    samples = []
    for sample in data.get("samples", []):
        snapshot_path = sample.get("snapshot_path")
        if not snapshot_path:
            continue
        image = cv2.imread(str(resolve_project_path(snapshot_path)))
        if image is None:
            continue
        color = extract_uniform_color_from_crop(deps, image)
        if color is not None:
            samples.append(color)
        if len(samples) >= limit:
            break
    if samples:
        print(f"Loaded uniform color samples from embeddings: {len(samples)}")
    else:
        print("Warning: no uniform color samples found in embeddings; falling back to video warmup.")
    return samples


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def configure_embedding_backend(deps: dict[str, Any], embedding_info: dict[str, Any], device: str, enabled: bool = True) -> None:
    embedding_type = str(embedding_info.get("type", ""))
    deps["embedding_type"] = embedding_type
    if not enabled:
        print("Player ReID disabled; using OCR, tracking, and position prediction only.")
        deps["reid_embedder"] = None
        return
    if embedding_type.startswith("torchreid_osnet"):
        print(f"Loading player ReID embedder: {embedding_type} on {device}")
        checkpoint = embedding_info.get("checkpoint") if "soccernet-osnet" in embedding_type else None
        deps["reid_embedder"] = OSNetEmbedder(
            device=device,
            checkpoint_path=checkpoint,
            pretrained=checkpoint is None,
        )
    else:
        deps["reid_embedder"] = None


def configure_ocr_backend(
    deps: dict[str, Any],
    args: argparse.Namespace,
    embedding_info: dict[str, Any],
    player_refs: list[dict[str, Any]],
) -> None:
    deps["jersey_ocr"] = None
    if args.ocr == "off":
        return
    if not any(ref.get("jersey_number") or ref.get("shirt_name") for ref in player_refs):
        print("Shirt OCR disabled: no jersey numbers or shirt names saved in player embeddings.")
        return
    try:
        print(f"Loading jersey OCR on {args.ocr_device}")
        deps["jersey_ocr"] = JerseyOCR(
            gpu=args.ocr_device == "cuda",
            min_confidence=args.ocr_min_confidence,
            languages=parse_languages(args.ocr_languages),
            backend=args.ocr_backend,
            model_dir=f"external/{args.ocr_backend}",
        )
    except Exception as exc:  # noqa: BLE001 - OCR is optional; ReID is the fallback.
        print(f"Warning: jersey OCR unavailable, using ReID embeddings only: {exc}")


def should_run_ocr(args: argparse.Namespace, frame_idx: int) -> bool:
    if args.ocr == "off":
        return False
    every = max(1, args.ocr_every_n_frames)
    return frame_idx % every == 0


def parse_languages(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()] or ["en"]


def merge_roster_refs(player_refs: list[dict[str, Any]], roster: PlayerRoster) -> list[dict[str, Any]]:
    refs_by_id = {ref["player_id"]: dict(ref) for ref in player_refs}
    for player in roster.players:
        ref = refs_by_id.setdefault(player.player_id, {"player_id": player.player_id, "embedding": []})
        if player.jersey_number:
            ref["jersey_number"] = player.jersey_number
        if player.names:
            ref["shirt_name"] = player.names[0]
            ref["shirt_names"] = player.names
    return list(refs_by_id.values())


def create_player_tracker(args: argparse.Namespace) -> Any:
    if args.tracker == "iou":
        return None
    if args.tracker == "bytetrack":
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker  # type: ignore
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing package for ByteTrack: lap\n"
                "Install with: ./venv/bin/python -m pip install lap"
            ) from exc
        return BYTETracker(
            argparse.Namespace(
                track_high_thresh=args.bytetrack_high_thresh,
                track_low_thresh=args.bytetrack_low_thresh,
                new_track_thresh=args.bytetrack_new_track_thresh,
                match_thresh=args.bytetrack_match_thresh,
                track_buffer=args.bytetrack_track_buffer,
                fuse_score=True,
            )
        )
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing package: deep-sort-realtime\n"
            "Install with: ./venv/bin/python -m pip install deep-sort-realtime"
        ) from exc

    return DeepSort(
        max_age=args.deepsort_max_age,
        n_init=args.deepsort_n_init,
        max_cosine_distance=args.deepsort_max_cosine_distance,
        nms_max_overlap=1.0,
        embedder=None,
    )


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
    if mode == "court-intersection":
        return [candidate for candidate in candidates if candidate_intersects_polygon(candidate, polygon)]
    if mode == "court-nearest-6":
        return nearest_court_candidates(candidates, polygon, limit=6)
    if mode == "lower-half":
        return [candidate for candidate in candidates if candidate[3] >= height * 0.48]
    if mode == "largest":
        return largest_candidates(candidates)

    intersection_hits = [candidate for candidate in candidates if candidate_intersects_polygon(candidate, polygon)]
    if intersection_hits:
        return intersection_hits
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


def suppress_duplicate_person_boxes(
    candidates: list[tuple[float, float, float, float, float]],
    threshold: float,
) -> list[tuple[float, float, float, float, float]]:
    if threshold <= 0:
        return candidates
    kept: list[tuple[float, float, float, float, float]] = []
    for candidate in sorted(candidates, key=lambda item: item[4], reverse=True):
        candidate_box = candidate[:4]
        if any(iou(candidate_box, kept_candidate[:4]) >= threshold for kept_candidate in kept):
            continue
        kept.append(candidate)
    return kept


def extract_uniform_color(
    deps: dict[str, Any],
    frame: Any,
    candidate: tuple[float, float, float, float, float],
) -> tuple[float, float, float] | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2, _confidence = candidate

    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    torso_x1 = int(max(0, min(width - 1, x1 + box_width * 0.18)))
    torso_x2 = int(max(0, min(width, x2 - box_width * 0.18)))
    torso_y1 = int(max(0, min(height - 1, y1 + box_height * 0.18)))
    torso_y2 = int(max(0, min(height, y1 + box_height * 0.62)))
    if torso_x2 <= torso_x1 or torso_y2 <= torso_y1:
        return None

    crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
    if crop.size == 0:
        return None
    return extract_uniform_color_from_crop(deps, crop)


def extract_uniform_color_from_crop(
    deps: dict[str, Any],
    crop: Any,
) -> tuple[float, float, float] | None:
    cv2 = deps["cv2"]
    np = deps["np"]
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = (saturation >= 25) & (value >= 35) & (value <= 245)
    pixels = hsv[mask]
    if len(pixels) < 24:
        pixels = hsv.reshape(-1, 3)
    if len(pixels) == 0:
        return None
    median = np.median(pixels.astype("float32"), axis=0)
    return (float(median[0]), float(median[1]), float(median[2]))


def median_hsv(samples: list[tuple[float, float, float]], np: Any) -> tuple[float, float, float]:
    values = np.asarray(samples, dtype="float32")
    hue_radians = values[:, 0] / 180.0 * 2.0 * np.pi
    mean_sin = float(np.mean(np.sin(hue_radians)))
    mean_cos = float(np.mean(np.cos(hue_radians)))
    hue = (np.arctan2(mean_sin, mean_cos) % (2.0 * np.pi)) / (2.0 * np.pi) * 180.0
    saturation = float(np.median(values[:, 1]))
    value = float(np.median(values[:, 2]))
    return (float(hue), saturation, value)


def hsv_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    hue_delta = abs(a[0] - b[0])
    hue_delta = min(hue_delta, 180.0 - hue_delta)
    saturation_delta = a[1] - b[1]
    value_delta = a[2] - b[2]
    return math.sqrt((2.0 * hue_delta) ** 2 + saturation_delta**2 + (0.45 * value_delta) ** 2)


def format_hsv(value: tuple[float, float, float]) -> str:
    return f"({value[0]:.1f}, {value[1]:.1f}, {value[2]:.1f})"


def nearest_court_candidates(
    candidates: list[tuple[float, float, float, float, float]],
    polygon: list[tuple[float, float]],
    limit: int,
) -> list[tuple[float, float, float, float, float]]:
    court_hits = [candidate for candidate in candidates if candidate_intersects_polygon(candidate, polygon)]
    if len(polygon) < 2:
        return largest_candidates(court_hits or candidates, limit=limit)
    front_a, front_b = polygon[0], polygon[1]
    court_min_y = min(point[1] for point in polygon)

    def candidate_tier(candidate: tuple[float, float, float, float, float]) -> int:
        if candidate in court_hits:
            return 0
        if candidate_foot(candidate)[1] >= court_min_y:
            return 1
        return 2

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate_tier(candidate),
            distance_point_to_segment(candidate_foot(candidate), front_a, front_b),
            -candidate[4],
        ),
    )
    return ranked[:limit]


def candidate_foot(candidate: tuple[float, float, float, float, float]) -> tuple[float, float]:
    x1, _, x2, y2, _ = candidate
    return ((x1 + x2) / 2, y2)


def candidate_in_polygon(
    candidate: tuple[float, float, float, float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    x1, y1, x2, y2, _ = candidate
    foot = ((x1 + x2) / 2, y2)
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    return point_in_polygon(foot, polygon) or point_in_polygon(center, polygon)


def candidate_intersects_polygon(
    candidate: tuple[float, float, float, float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    if not polygon:
        return True
    x1, y1, x2, y2, _ = candidate
    bbox_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if any(point_in_polygon(point, polygon) for point in bbox_corners):
        return True
    if any(point_inside_bbox(point, (x1, y1, x2, y2)) for point in polygon):
        return True

    bbox_edges = list(zip(bbox_corners, [*bbox_corners[1:], bbox_corners[0]]))
    polygon_edges = list(zip(polygon, [*polygon[1:], polygon[0]]))
    return any(
        segments_intersect(bbox_a, bbox_b, poly_a, poly_b)
        for bbox_a, bbox_b in bbox_edges
        for poly_a, poly_b in polygon_edges
    )


def build_player_detections(
    deps: dict[str, Any],
    frame: Any,
    frame_idx: int,
    candidates: list[tuple[float, float, float, float, float]],
    player_refs: list[dict[str, Any]],
    threshold: float,
    ocr_enabled: bool,
    ocr_overlap_iou: float,
    roster: PlayerRoster | None,
    use_reid: bool,
) -> list[PlayerDetection]:
    detections = []
    for idx, (x1, y1, x2, y2, confidence) in enumerate(candidates):
        crop = frame[max(0, int(y1)) : max(0, int(y2)), max(0, int(x1)) : max(0, int(x2))]
        if crop.size == 0:
            continue
        embedding = compute_crop_embedding(deps, crop)
        can_read_ocr = ocr_enabled and not candidate_overlaps_other(candidates, idx, ocr_overlap_iou)
        name_identity, number_identity = read_crop_identities(deps, crop, can_read_ocr)
        jersey_number = number_identity.value if number_identity is not None else None
        jersey_confidence = number_identity.confidence if number_identity is not None else None
        shirt_name = name_identity.value if name_identity is not None else None
        shirt_name_confidence = name_identity.confidence if name_identity is not None else None
        detections.append(
            PlayerDetection(
                frame=frame_idx,
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                embedding=embedding,
                player_id=None,
                player_distance=None,
                jersey_number=jersey_number,
                jersey_confidence=jersey_confidence,
                shirt_name=shirt_name,
                shirt_name_confidence=shirt_name_confidence,
                identity_source=None,
            )
        )
    assign_players_to_detections(detections, player_refs, threshold, roster, use_reid)
    return detections


def candidate_overlaps_other(
    candidates: list[tuple[float, float, float, float, float]],
    index: int,
    threshold: float,
) -> bool:
    if threshold <= 0:
        return False
    bbox = candidates[index][:4]
    for other_idx, other in enumerate(candidates):
        if other_idx == index:
            continue
        if iou(bbox, other[:4]) >= threshold:
            return True
    return False


def assign_players_to_detections(
    detections: list[PlayerDetection],
    player_refs: list[dict[str, Any]],
    threshold: float,
    roster: PlayerRoster | None = None,
    use_reid: bool = True,
) -> None:
    for detection in detections:
        detection.player_id = None
        detection.player_distance = None
        detection.identity_source = None

    assign_ocr_players_to_detections(detections, player_refs, roster)
    if use_reid:
        assign_reid_players_to_detections(detections, player_refs, threshold)


def assign_ocr_players_to_detections(
    detections: list[PlayerDetection],
    player_refs: list[dict[str, Any]],
    roster: PlayerRoster | None = None,
) -> None:
    if roster is not None:
        assign_roster_ocr_players_to_detections(detections, roster)
        return

    refs_by_number: dict[str, list[dict[str, Any]]] = {}
    refs_by_name: dict[str, list[dict[str, Any]]] = {}
    for ref in player_refs:
        jersey_number = ref.get("jersey_number")
        if jersey_number:
            refs_by_number.setdefault(jersey_number, []).append(ref)
        shirt_name = ref.get("shirt_name")
        if shirt_name:
            refs_by_name.setdefault(shirt_name, []).append(ref)

    candidates = sorted(
        detections,
        key=lambda item: max(item.shirt_name_confidence or 0.0, item.jersey_confidence or 0.0),
        reverse=True,
    )
    used_players = set()
    for detection in candidates:
        if detection.shirt_name:
            refs = refs_by_name.get(detection.shirt_name, [])
        elif detection.jersey_number:
            refs = refs_by_number.get(detection.jersey_number, [])
        else:
            continue
        if len(refs) != 1:
            continue
        player_id = refs[0]["player_id"]
        if player_id in used_players:
            continue
        detection.player_id = player_id
        detection.player_distance = 0.0
        detection.identity_source = "ocr_name" if detection.shirt_name else "ocr_number"
        used_players.add(player_id)


def assign_roster_ocr_players_to_detections(
    detections: list[PlayerDetection],
    roster: PlayerRoster,
) -> None:
    candidates = sorted(
        detections,
        key=lambda item: max(item.shirt_name_confidence or 0.0, item.jersey_confidence or 0.0),
        reverse=True,
    )
    used_players = set()
    for detection in candidates:
        match = match_detection_with_roster_ocr(detection, roster)
        if match is None:
            continue
        if match.player.player_id in used_players:
            continue
        detection.player_id = match.player.player_id
        detection.player_distance = 0.0
        detection.identity_source = "ocr_name" if match.kind == "name" else "ocr_number"
        if match.player.jersey_number:
            detection.jersey_number = match.player.jersey_number
        if match.player.names:
            detection.shirt_name = match.player.names[0]
        used_players.add(match.player.player_id)


def match_detection_with_roster_ocr(detection: PlayerDetection, roster: PlayerRoster) -> Any | None:
    if detection.shirt_name:
        identity = OCRIdentity("name", detection.shirt_name, float(detection.shirt_name_confidence or 0.0), detection.shirt_name)
        match = roster.match_identity(identity)
        if match is None:
            return None
        roster_number = match.player.jersey_number
        if roster_number:
            if not detection.jersey_number or normalize_jersey_number(detection.jersey_number) != roster_number:
                return None
        return match
    if detection.jersey_number:
        identity = OCRIdentity("number", detection.jersey_number, float(detection.jersey_confidence or 0.0), detection.jersey_number)
        return roster.match_identity(identity)
    return None


def assign_reid_players_to_detections(
    detections: list[PlayerDetection],
    player_refs: list[dict[str, Any]],
    threshold: float,
) -> None:
    used_players = {detection.player_id for detection in detections if detection.player_id}
    used_detections = {id(detection) for detection in detections if detection.player_id}

    pairs = []
    for ref in player_refs:
        if not ref.get("embedding"):
            continue
        for detection in detections:
            if id(detection) in used_detections:
                continue
            pairs.append((euclidean(detection.embedding, ref["embedding"]), ref["player_id"], detection))
    pairs.sort(key=lambda item: item[0])

    for distance, player_id, detection in pairs:
        if threshold > 0 and distance > threshold:
            continue
        if player_id in used_players or id(detection) in used_detections:
            continue
        detection.player_id = player_id
        detection.player_distance = distance
        detection.identity_source = "reid"
        used_players.add(player_id)
        used_detections.add(id(detection))


class ByteTrackResults:
    def __init__(self, detections: list[PlayerDetection]) -> None:
        import numpy as np  # type: ignore

        self._np = np
        if detections:
            self.xyxy = np.asarray([detection.bbox for detection in detections], dtype=np.float32)
            self.conf = np.asarray([detection.confidence for detection in detections], dtype=np.float32)
            self.cls = np.zeros(len(detections), dtype=np.float32)
        else:
            self.xyxy = np.empty((0, 4), dtype=np.float32)
            self.conf = np.empty((0,), dtype=np.float32)
            self.cls = np.empty((0,), dtype=np.float32)
        self.xywh = xyxy_to_xywh(self.xyxy)

    def __len__(self) -> int:
        return int(len(self.conf))

    def __getitem__(self, item: Any) -> "ByteTrackResults":
        result = object.__new__(ByteTrackResults)
        result._np = self._np
        result.xyxy = self.xyxy[item]
        result.conf = self.conf[item]
        result.cls = self.cls[item]
        if result.xyxy.ndim == 1:
            result.xyxy = result.xyxy.reshape(1, 4)
            result.conf = result.conf.reshape(1)
            result.cls = result.cls.reshape(1)
        result.xywh = xyxy_to_xywh(result.xyxy)
        return result


def xyxy_to_xywh(xyxy: Any) -> Any:
    xywh = xyxy.copy()
    if len(xywh) == 0:
        return xywh
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return xywh


def update_bytetrack_player_tracks(
    tracker: Any,
    player_tracks: list[PlayerTrack],
    detections: list[PlayerDetection],
    frame_idx: int,
    frame: Any,
    max_missed: int,
    ocr_relabel_min_confidence: float,
) -> None:
    results = tracker.update(ByteTrackResults(detections), img=frame)
    active_ids = set()
    for row in results:
        if len(row) < 8:
            continue
        x1, y1, x2, y2 = [float(value) for value in row[:4]]
        track_id = int(row[4])
        score = float(row[5])
        source_idx = int(row[7])
        source_detection = detections[source_idx] if 0 <= source_idx < len(detections) else None
        existing_track = find_player_track(player_tracks, track_id)
        identity = choose_stable_track_identity(
            source_detection,
            existing_track,
            frame_idx,
            ocr_relabel_min_confidence,
        )
        confidence = source_detection.confidence if source_detection is not None else score

        detection = PlayerDetection(
            frame=frame_idx,
            bbox=(x1, y1, x2, y2),
            confidence=confidence,
            embedding=identity["embedding"],
            player_id=identity["player_id"],
            player_distance=identity["player_distance"],
            jersey_number=identity["jersey_number"],
            jersey_confidence=identity["jersey_confidence"],
            shirt_name=identity["shirt_name"],
            shirt_name_confidence=identity["shirt_name_confidence"],
            identity_source=identity["identity_source"],
        )
        if existing_track is None:
            existing_track = PlayerTrack(track_id=track_id, detections=[detection], missed=0)
            player_tracks.append(existing_track)
        else:
            if not existing_track.detections or existing_track.detections[-1].frame != frame_idx:
                existing_track.detections.append(detection)
            else:
                existing_track.detections[-1] = detection
            existing_track.missed = 0
        active_ids.add(track_id)

    for player_track in player_tracks:
        if player_track.track_id not in active_ids:
            player_track.missed += 1
    player_tracks[:] = [track for track in player_tracks if track.missed <= max_missed]


def update_deepsort_player_tracks(
    tracker: Any,
    player_tracks: list[PlayerTrack],
    detections: list[PlayerDetection],
    frame_idx: int,
    max_missed: int,
    ocr_relabel_min_confidence: float,
) -> None:
    raw_detections = []
    embeds = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        raw_detections.append(([x1, y1, x2 - x1, y2 - y1], detection.confidence, "player"))
        embeds.append(detection.embedding)

    tracks = tracker.update_tracks(raw_detections, embeds=embeds, others=detections)
    active_ids = set()
    for track in tracks:
        if track.is_deleted():
            continue
        ltrb = track.to_ltrb(orig=False)
        if ltrb is None:
            continue
        x1, y1, x2, y2 = [float(value) for value in ltrb]
        source_detection = track.get_det_supplementary()
        existing_track = find_player_track(player_tracks, track.track_id)
        confidence = 0.0
        if isinstance(source_detection, PlayerDetection):
            confidence = source_detection.confidence
        elif existing_track and existing_track.detections:
            confidence = existing_track.detections[-1].confidence
        identity = choose_stable_track_identity(
            source_detection if isinstance(source_detection, PlayerDetection) else None,
            existing_track,
            frame_idx,
            ocr_relabel_min_confidence,
        )

        detection = PlayerDetection(
            frame=frame_idx,
            bbox=(x1, y1, x2, y2),
            confidence=confidence,
            embedding=identity["embedding"],
            player_id=identity["player_id"],
            player_distance=identity["player_distance"],
            jersey_number=identity["jersey_number"],
            jersey_confidence=identity["jersey_confidence"],
            shirt_name=identity["shirt_name"],
            shirt_name_confidence=identity["shirt_name_confidence"],
            identity_source=identity["identity_source"],
        )
        if existing_track is None:
            existing_track = PlayerTrack(track_id=track.track_id, detections=[detection], missed=0)
            player_tracks.append(existing_track)
        else:
            if not existing_track.detections or existing_track.detections[-1].frame != frame_idx:
                existing_track.detections.append(detection)
            else:
                existing_track.detections[-1] = detection
            existing_track.missed = 0
        active_ids.add(track.track_id)

    for player_track in player_tracks:
        if player_track.track_id not in active_ids:
            player_track.missed += 1
    player_tracks[:] = [track for track in player_tracks if track.missed <= max_missed]


def choose_stable_track_identity(
    source_detection: PlayerDetection | None,
    existing_track: PlayerTrack | None,
    frame_idx: int,
    ocr_relabel_min_confidence: float,
) -> dict[str, Any]:
    previous = existing_track.detections[-1] if existing_track and existing_track.detections else None

    # Tracking is the default authority once a label is attached to a track.
    # A strong roster OCR match may correct an existing track label, then
    # prediction carries that corrected identity through detector misses.
    if previous is not None and previous.player_id:
        if (
            source_detection is not None
            and source_detection.identity_source in {"ocr_name", "ocr_number"}
            and source_detection.player_id
            and source_detection.player_id != previous.player_id
            and ocr_relabel_is_allowed(
                source_detection,
                existing_track,
                frame_idx,
                ocr_relabel_min_confidence,
            )
        ):
            return detection_identity(source_detection)
        identity = detection_identity(previous)
        if source_detection is not None:
            identity["embedding"] = source_detection.embedding
            identity["jersey_number"] = source_detection.jersey_number or identity["jersey_number"]
            identity["jersey_confidence"] = source_detection.jersey_confidence or identity["jersey_confidence"]
            identity["shirt_name"] = source_detection.shirt_name or identity["shirt_name"]
            identity["shirt_name_confidence"] = source_detection.shirt_name_confidence or identity["shirt_name_confidence"]
        identity["identity_source"] = "track"
        return identity

    if source_detection is not None and source_detection.player_id:
        return detection_identity(source_detection)
    if previous is not None:
        return detection_identity(previous)
    return {
        "embedding": source_detection.embedding if source_detection is not None else [],
        "player_id": None,
        "player_distance": None,
        "jersey_number": None,
        "jersey_confidence": None,
        "shirt_name": None,
        "shirt_name_confidence": None,
        "identity_source": None,
    }


def ocr_relabel_is_allowed(
    source_detection: PlayerDetection,
    existing_track: PlayerTrack | None,
    frame_idx: int,
    min_confidence: float,
) -> bool:
    confidence = ocr_identity_confidence(source_detection)
    return confidence >= min_confidence


def ocr_identity_confidence(detection: PlayerDetection) -> float:
    if detection.identity_source == "ocr_name":
        return float(detection.shirt_name_confidence or 0.0)
    if detection.identity_source == "ocr_number":
        return float(detection.jersey_confidence or 0.0)
    return 0.0


def detection_identity(detection: PlayerDetection) -> dict[str, Any]:
    return {
        "embedding": detection.embedding,
        "player_id": detection.player_id,
        "player_distance": detection.player_distance,
        "jersey_number": detection.jersey_number,
        "jersey_confidence": detection.jersey_confidence,
        "shirt_name": detection.shirt_name,
        "shirt_name_confidence": detection.shirt_name_confidence,
        "identity_source": detection.identity_source,
    }


def find_player_track(player_tracks: list[PlayerTrack], track_id: int | str) -> PlayerTrack | None:
    for player_track in player_tracks:
        if player_track.track_id == track_id:
            return player_track
    return None


def current_tracker_detections(player_tracks: list[PlayerTrack], frame_idx: int) -> list[PlayerDetection]:
    detections = []
    for track in player_tracks:
        if track.detections and track.detections[-1].frame == frame_idx:
            detections.append(track.detections[-1])
    return detections


def max_tracker_missed(args: argparse.Namespace) -> int:
    if args.predict_missing_players:
        return max(2, args.max_player_prediction_gap)
    return 2


def predict_missing_roster_tracks(
    player_tracks: list[PlayerTrack],
    frame_idx: int,
    roster: PlayerRoster | None,
    width: int,
    height: int,
    enabled: bool,
) -> None:
    if not enabled or roster is None:
        return

    current_ids = {
        track.detections[-1].player_id
        for track in player_tracks
        if track.detections and track.detections[-1].frame == frame_idx and track.detections[-1].player_id
    }
    roster_ids = {player.player_id for player in roster.players}
    missing_ids = roster_ids - current_ids
    if not missing_ids:
        return

    tracks_by_player: dict[str, PlayerTrack] = {}
    for track in player_tracks:
        if not track.detections:
            continue
        last = track.detections[-1]
        if last.frame == frame_idx or last.player_id not in missing_ids:
            continue
        current = tracks_by_player.get(last.player_id)
        if current is None or last.frame > current.detections[-1].frame:
            tracks_by_player[last.player_id] = track

    for player_id, track in tracks_by_player.items():
        last = track.detections[-1]
        gap = frame_idx - last.frame
        if gap <= 0:
            continue
        vx, vy = estimate_track_velocity(track)
        predicted_bbox = shift_bbox(last.bbox, vx * gap, vy * gap, width, height)
        player = roster.player_by_id(player_id)
        predicted = PlayerDetection(
            frame=frame_idx,
            bbox=predicted_bbox,
            confidence=max(0.05, last.confidence * (0.90**gap)),
            embedding=last.embedding,
            player_id=last.player_id,
            player_distance=last.player_distance,
            jersey_number=last.jersey_number or (player.jersey_number if player else None),
            jersey_confidence=last.jersey_confidence,
            shirt_name=last.shirt_name or (player.names[0] if player and player.names else None),
            shirt_name_confidence=last.shirt_name_confidence,
            identity_source="predicted_track",
        )
        track.detections.append(predicted)


def estimate_track_velocity(track: PlayerTrack) -> tuple[float, float]:
    usable = [
        detection
        for detection in track.detections
        if detection.identity_source != "predicted_track"
    ]
    if len(usable) < 2:
        usable = track.detections
    if len(usable) < 2:
        return (0.0, 0.0)

    recent = usable[-4:]
    velocities = []
    for prev, cur in zip(recent, recent[1:]):
        dt = max(1, cur.frame - prev.frame)
        px, py = bbox_center(prev.bbox)
        cx, cy = bbox_center(cur.bbox)
        velocities.append(((cx - px) / dt, (cy - py) / dt))
    if not velocities:
        return (0.0, 0.0)
    vx = sum(item[0] for item in velocities) / len(velocities)
    vy = sum(item[1] for item in velocities) / len(velocities)
    speed = math.hypot(vx, vy)
    max_speed = 45.0
    if speed > max_speed:
        scale = max_speed / speed
        vx *= scale
        vy *= scale
    return (vx, vy)


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def shift_bbox(
    bbox: tuple[float, float, float, float],
    dx: float,
    dy: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    new_x1 = min(max(0.0, x1 + dx), max(0.0, width - box_width))
    new_y1 = min(max(0.0, y1 + dy), max(0.0, height - box_height))
    return (new_x1, new_y1, new_x1 + box_width, new_y1 + box_height)


def fill_current_roster_labels(
    player_tracks: list[PlayerTrack],
    frame_idx: int,
    roster: PlayerRoster | None,
    player_refs: list[dict[str, Any]],
    enabled: bool,
    use_reid: bool = True,
    reid_max_center_jump: float = 0.0,
) -> None:
    if not enabled or roster is None:
        return

    current = [
        (track, track.detections[-1])
        for track in player_tracks
        if track.detections and track.detections[-1].frame == frame_idx
    ]
    if not current:
        return

    roster_ids = [player.player_id for player in roster.players]
    roster_id_set = set(roster_ids)
    clear_duplicate_current_labels(current)

    used_ids = {
        detection.player_id
        for _track, detection in current
        if detection.player_id in roster_id_set
    }
    missing_ids = [player_id for player_id in roster_ids if player_id not in used_ids]
    unlabeled = [item for item in current if item[1].player_id not in roster_id_set]
    if not missing_ids or not unlabeled:
        return

    refs_by_id = {ref["player_id"]: ref for ref in player_refs if ref.get("embedding")} if use_reid else {}
    expected_centers = expected_roster_centers(player_tracks, frame_idx, missing_ids)
    assignments = choose_roster_fill_assignments(
        unlabeled,
        missing_ids,
        refs_by_id,
        expected_centers,
        reid_max_center_jump,
    )
    for (_track, detection), player_id, distance in assignments:
        player = roster.player_by_id(player_id)
        detection.player_id = player_id
        detection.player_distance = distance
        detection.identity_source = "roster_fill_reid" if distance is not None else "roster_fill_position"
        if player and player.jersey_number:
            detection.jersey_number = player.jersey_number
            detection.jersey_confidence = detection.jersey_confidence or 0.0
        if player and player.names:
            detection.shirt_name = player.names[0]


def clear_duplicate_current_labels(
    current: list[tuple[PlayerTrack, PlayerDetection]],
) -> None:
    grouped: dict[str, list[tuple[PlayerTrack, PlayerDetection]]] = {}
    for track, detection in current:
        if detection.player_id:
            grouped.setdefault(detection.player_id, []).append((track, detection))

    for player_id, items in grouped.items():
        if len(items) <= 1:
            continue
        keep_track, keep_detection = min(
            items,
            key=lambda item: (
                identity_source_rank(item[1].identity_source),
                item[1].player_distance if item[1].player_distance is not None else float("inf"),
            ),
        )
        for track, detection in items:
            if track is keep_track and detection is keep_detection:
                continue
            detection.player_id = None
            detection.player_distance = None
            detection.identity_source = None


def identity_source_rank(source: str | None) -> int:
    order = {
        "ocr_name": 0,
        "ocr_number": 1,
        "track": 2,
        "reid": 3,
        "roster_fill_reid": 4,
        "roster_fill_position": 5,
        "predicted_track": 6,
    }
    return order.get(source or "", 9)


def choose_roster_fill_assignments(
    unlabeled: list[tuple[PlayerTrack, PlayerDetection]],
    missing_ids: list[str],
    refs_by_id: dict[str, dict[str, Any]],
    expected_centers: dict[str, tuple[float, float]],
    max_center_jump: float,
) -> list[tuple[tuple[PlayerTrack, PlayerDetection], str, float | None]]:
    pairs = []
    for item in unlabeled:
        detection = item[1]
        for player_id in missing_ids:
            ref = refs_by_id.get(player_id)
            if not ref:
                continue
            if not reid_roster_fill_is_allowed(detection, player_id, expected_centers, max_center_jump):
                continue
            pairs.append((euclidean(detection.embedding, ref["embedding"]), item, player_id))
    pairs.sort(key=lambda value: value[0])

    assignments: list[tuple[tuple[PlayerTrack, PlayerDetection], str, float | None]] = []
    used_items = set()
    used_players = set()
    for distance, item, player_id in pairs:
        item_key = id(item[1])
        if item_key in used_items or player_id in used_players:
            continue
        assignments.append((item, player_id, distance))
        used_items.add(item_key)
        used_players.add(player_id)

    remaining_items = [item for item in unlabeled if id(item[1]) not in used_items]
    remaining_players = [player_id for player_id in missing_ids if player_id not in used_players]
    if remaining_items and remaining_players:
        remaining_items.sort(key=lambda item: bbox_center_x(item[1].bbox))
        for item, player_id in zip(remaining_items, remaining_players):
            if player_id in expected_centers and max_center_jump > 0:
                continue
            assignments.append((item, player_id, None))

    return assignments


def expected_roster_centers(
    player_tracks: list[PlayerTrack],
    frame_idx: int,
    player_ids: list[str],
) -> dict[str, tuple[float, float]]:
    player_id_set = set(player_ids)
    expected: dict[str, tuple[float, float]] = {}
    latest_frame: dict[str, int] = {}
    for track in player_tracks:
        if not track.detections:
            continue
        last = track.detections[-1]
        if last.player_id not in player_id_set or last.frame >= frame_idx:
            continue
        gap = max(1, frame_idx - last.frame)
        vx, vy = estimate_track_velocity(track)
        center = bbox_center(last.bbox)
        predicted = (center[0] + vx * gap, center[1] + vy * gap)
        if last.player_id not in expected or last.frame > latest_frame[last.player_id]:
            expected[last.player_id] = predicted
            latest_frame[last.player_id] = last.frame
    return expected


def reid_roster_fill_is_allowed(
    detection: PlayerDetection,
    player_id: str,
    expected_centers: dict[str, tuple[float, float]],
    max_center_jump: float,
) -> bool:
    if max_center_jump <= 0:
        return True
    expected = expected_centers.get(player_id)
    if expected is None:
        return True
    observed = bbox_center(detection.bbox)
    return math.hypot(observed[0] - expected[0], observed[1] - expected[1]) <= max_center_jump


def bbox_center_x(bbox: tuple[float, float, float, float]) -> float:
    return (bbox[0] + bbox[2]) / 2.0


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
    max_jump: float,
    reacquire_gap: int,
    reacquire_max_jump: float,
    reset_gap: int,
) -> BallObservation | None:
    if state.missed >= max(1, reset_gap):
        state.last = None
        state.prev = None
        state.vx = 0.0
        state.vy = 0.0
        state.missed = 0

    predicted = state.predict(frame_idx)
    effective_max_jump = max_jump
    if state.missed >= max(1, reacquire_gap):
        effective_max_jump = reacquire_max_jump
    accepted = choose_ball_candidate(
        candidates,
        predicted,
        has_track=state.last is not None,
        max_jump=effective_max_jump,
    )
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
    max_jump: float,
) -> BallObservation | None:
    if not candidates:
        return None
    if not has_track or predicted is None:
        best_initial = max(candidates, key=lambda candidate: candidate.confidence)
        return best_initial if best_initial.confidence >= 0.34 else None

    def score(candidate: BallObservation) -> float:
        distance = math.hypot(candidate.x - predicted.x, candidate.y - predicted.y)
        dt = max(1, candidate.frame - predicted.frame)
        expected_limit = max_jump * dt
        if distance > expected_limit:
            return -float("inf")
        distance_penalty = min(1.0, distance / expected_limit)
        source_bonus = 0.18 if candidate.source == "vball-net" else 0.12 if candidate.source == "yolo_sports_ball" else 0.0
        return candidate.confidence + source_bonus - 0.65 * distance_penalty

    best = max(candidates, key=score)
    if score(best) < 0.02:
        return None
    return best


def update_player_tracks(
    tracks: list[PlayerTrack],
    detections: list[PlayerDetection],
    next_track_id: int,
    max_missed: int,
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
    tracks[:] = [track for track in tracks if track.missed <= max_missed or len(track.detections) >= 2]
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
        if use_player_label and det.identity_source == "ocr_name":
            label = f"{label} {det.shirt_name}"
        elif use_player_label and det.identity_source == "ocr_number":
            label = f"{label} #{det.jersey_number}"
        if use_player_label:
            ocr_confidence = display_ocr_confidence(det)
            if ocr_confidence is not None:
                label = f"{label} ocr={ocr_confidence:.2f}"
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
        if detection.player_id is None:
            continue
        current_best = best_by_player.get(detection.player_id)
        if current_best is None or visible_detection_rank(detection) < visible_detection_rank(current_best):
            best_by_player[detection.player_id] = detection
    return {id(detection) for detection in best_by_player.values()}


def display_ocr_confidence(detection: PlayerDetection) -> float | None:
    values = [
        value
        for value in (detection.jersey_confidence, detection.shirt_name_confidence)
        if value is not None
    ]
    if not values:
        return None
    return float(max(values))


def visible_detection_rank(detection: PlayerDetection) -> tuple[int, float]:
    distance = detection.player_distance if detection.player_distance is not None else float("inf")
    return (identity_source_rank(detection.identity_source), distance)


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


def read_crop_identities(
    deps: dict[str, Any],
    crop: Any,
    enabled: bool,
) -> tuple[OCRIdentity | None, OCRIdentity | None]:
    ocr = deps.get("jersey_ocr")
    if not enabled or ocr is None:
        return None, None
    name, name_confidence = ocr.read_name(deps["cv2"], crop)
    number, number_confidence = ocr.read_number(deps["cv2"], crop)
    name_identity = (
        OCRIdentity("name", name, float(name_confidence), name)
        if name and name_confidence is not None
        else None
    )
    number_identity = (
        OCRIdentity("number", number, float(number_confidence), number)
        if number and number_confidence is not None
        else None
    )
    return name_identity, number_identity


def match_player(
    embedding: list[float],
    refs: list[dict[str, Any]],
    threshold: float,
) -> tuple[str | None, float | None]:
    best_id = None
    best_distance = float("inf")
    for ref in refs:
        if not ref.get("embedding"):
            continue
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
                    "jersey_number": det.jersey_number,
                    "jersey_confidence": det.jersey_confidence,
                    "shirt_name": det.shirt_name,
                    "shirt_name_confidence": det.shirt_name_confidence,
                    "identity_source": det.identity_source,
                }
                for det in track.detections
            ],
        }
        for track in tracks
    ]


def serialize_visible_player_detections(
    tracks: list[PlayerTrack],
    frame: int,
    max_gap: int,
) -> dict[str, Any]:
    visible = []
    visible_detections = []
    for track in tracks:
        det = latest_detection_near_frame(track, frame, max_gap=max_gap)
        if det is None:
            continue
        visible_detections.append((track, det))

    unique_player_detection_ids = nearest_visible_detection_ids_by_player(visible_detections)
    for track, det in visible_detections:
        use_player_label = det.player_id is not None and id(det) in unique_player_detection_ids
        display_label = det.player_id if use_player_label else f"track_{track.track_id}"
        if use_player_label and det.identity_source == "ocr_name":
            display_label = f"{display_label} {det.shirt_name}"
        elif use_player_label and det.identity_source == "ocr_number":
            display_label = f"{display_label} #{det.jersey_number}"
        if use_player_label:
            ocr_confidence = display_ocr_confidence(det)
            if ocr_confidence is not None:
                display_label = f"{display_label} ocr={ocr_confidence:.2f}"

        visible.append(
            {
                "track_id": track.track_id,
                "source_frame": det.frame,
                "display_label": display_label,
                "uses_player_label": use_player_label,
                "bbox": {
                    "x1": det.bbox[0],
                    "y1": det.bbox[1],
                    "x2": det.bbox[2],
                    "y2": det.bbox[3],
                },
                "confidence": det.confidence,
                "player_id": det.player_id if use_player_label else None,
                "raw_player_id": det.player_id,
                "player_distance": det.player_distance,
                "jersey_number": det.jersey_number,
                "jersey_confidence": det.jersey_confidence,
                "shirt_name": det.shirt_name,
                "shirt_name_confidence": det.shirt_name_confidence,
                "identity_source": det.identity_source,
            }
        )
    return {"frame": frame, "detections": visible}


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
        "jersey_number",
        "jersey_confidence",
        "shirt_name",
        "shirt_name_confidence",
        "identity_source",
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
                    "jersey_number": "",
                    "jersey_confidence": "",
                    "shirt_name": "",
                    "shirt_name_confidence": "",
                    "identity_source": "",
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
                        "jersey_number": det.jersey_number or "",
                        "jersey_confidence": "" if det.jersey_confidence is None else round(det.jersey_confidence, 6),
                        "shirt_name": det.shirt_name or "",
                        "shirt_name_confidence": "" if det.shirt_name_confidence is None else round(det.shirt_name_confidence, 6),
                        "identity_source": det.identity_source or "",
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


def segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    def orientation(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    def on_segment(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> bool:
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    eps = 1e-9
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if abs(o1) <= eps and on_segment(a, c, b):
        return True
    if abs(o2) <= eps and on_segment(a, d, b):
        return True
    if abs(o3) <= eps and on_segment(c, a, d):
        return True
    if abs(o4) <= eps and on_segment(c, b, d):
        return True
    return False


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


def distance_point_to_segment(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    px, py = point
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


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
