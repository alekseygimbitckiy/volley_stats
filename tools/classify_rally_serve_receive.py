#!/usr/bin/env python3
"""Classify serve side and ball-contact actions in one-rally volleyball clips."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from temporal_args import resolve_temporal_option


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BallPoint:
    frame: int
    x: float
    y: float
    radius: float
    confidence: float
    tracklet_id: int | None = None


@dataclass
class PlayerBox:
    frame: int
    track_id: str
    player_id: str | None
    bbox: tuple[float, float, float, float]
    confidence: float | None
    team: str = "unknown"
    observed: bool = True

    @property
    def upper_anchor(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y1 + (y2 - y1) * 0.25)

    @property
    def court_anchor(self) -> tuple[float, float]:
        x1, _y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


@dataclass
class BallState:
    last: BallPoint | None = None
    vx: float = 0.0
    vy: float = 0.0
    missed: int = 0

    def predict(self, frame: int) -> BallPoint | None:
        if self.last is None:
            return None
        dt = max(1, frame - self.last.frame)
        return BallPoint(
            frame=frame,
            x=self.last.x + self.vx * dt,
            y=self.last.y + self.vy * dt,
            radius=self.last.radius,
            confidence=max(0.05, self.last.confidence * (0.72 ** dt)),
            tracklet_id=self.last.tracklet_id,
        )

    def update(self, point: BallPoint) -> None:
        if self.last is not None:
            dt = max(1, point.frame - self.last.frame)
            mx = (point.x - self.last.x) / dt
            my = (point.y - self.last.y) / dt
            alpha = 0.45
            self.vx = alpha * mx + (1 - alpha) * self.vx
            self.vy = alpha * my + (1 - alpha) * self.vy
        self.last = point
        self.missed = 0


@dataclass
class DirectionChangeMoment:
    point: BallPoint
    frame_float: float
    visible_before_frame: int
    visible_after_frame: int
    used_gap_midpoint: bool


def main() -> int:
    parser = argparse.ArgumentParser(description="Find serve side, ball-contact action moments, and receiving players.")
    parser.add_argument("video", help="Path to one rally video.")
    parser.add_argument("--ball-track", required=True, help="Normalized ball CSV/JSON.")
    parser.add_argument("--tracking-json", required=True, help="test_track_video.py JSON with player tracks.")
    parser.add_argument("--layout", default="data/processed/calibrations/field_layout.json")
    parser.add_argument("--reception-zones", default="data/processed/calibrations/reception_zones.json")
    parser.add_argument(
        "--team-filter",
        choices=["none", "classified-near", "court-intersection", "court-nearest-6"],
        default="classified-near",
        help="Filter player boxes from the tracking JSON before assigning actions.",
    )
    parser.add_argument("--output-dir", default="data/processed/rally_classification")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--serve-window-sec", type=float, default=14 / 30, help="Seconds used to score a smooth early serve flight.")
    parser.add_argument("--serve-window", dest="serve_window_frames", type=int, default=None, help="Deprecated frame override for --serve-window-sec.")
    parser.add_argument("--serve-search-ratio", type=float, default=0.55, help="Search serve in the first N ratio of the clip.")
    parser.add_argument("--serve-min-speed", type=float, default=8.0, help="Minimum average ball speed in px/frame for a serve flight.")
    parser.add_argument("--serve-min-distance", type=float, default=120.0, help="Minimum total ball displacement for a serve flight.")
    parser.add_argument("--serve-max-mean-angle-change", type=float, default=38.0, help="Max mean trajectory angle change inside a smooth serve flight.")
    parser.add_argument("--serve-early-bonus", type=float, default=0.35, help="How strongly to prefer earlier smooth serve flights.")
    parser.add_argument(
        "--serve-toss-search-sec",
        type=float,
        default=1.50,
        help="Seconds before serve flight searched for a compact near-vertical toss and the serve contact.",
    )
    parser.add_argument(
        "--serve-contact-max-gap-sec",
        type=float,
        default=0.33,
        help="Largest ball-visibility gap allowed between the last toss point/contact and the outbound serve flight.",
    )
    parser.add_argument(
        "--serve-contact-flight-sec",
        type=float,
        default=1.0,
        help=(
            "Seconds after a toss/contact candidate used to validate the outbound serve flight. "
            "Validation stops before the next strong direction change."
        ),
    )
    parser.add_argument(
        "--serve-contact-flight",
        dest="serve_contact_flight_frames",
        type=int,
        default=None,
        help="Deprecated frame override for --serve-contact-flight-sec.",
    )
    parser.add_argument(
        "--serve-contact-min-speed",
        type=float,
        default=4.0,
        help="Minimum average speed in px/frame for the longer post-contact serve validation window.",
    )
    parser.add_argument(
        "--serve-contact-player-max-distance",
        type=float,
        default=220.0,
        help="Maximum ball-to-player upper-body/wrist distance when identifying the server.",
    )
    parser.add_argument(
        "--serve-toss-max-x-drift",
        type=float,
        default=80.0,
        help="Maximum horizontal span in pixels for points considered part of the serve toss.",
    )
    parser.add_argument(
        "--serve-toss-min-vertical-motion",
        type=float,
        default=6.0,
        help="Minimum vertical toss motion in pixels needed to claim that serve contact is visible.",
    )
    parser.add_argument(
        "--serve-toss-min-vertical-x-ratio",
        type=float,
        default=2.0,
        help="Minimum ratio of vertical to horizontal toss motion; rejects ordinary diagonal serve flight as a toss.",
    )
    parser.add_argument(
        "--serve-power-toss-min-height",
        type=float,
        default=70.0,
        help="Vertical toss range in pixels that classifies a captured serve as a power serve.",
    )
    parser.add_argument(
        "--serve-power-min-speed",
        type=float,
        default=20.0,
        help="Minimum outbound speed used as supporting power-serve evidence when only part of the toss is visible.",
    )
    parser.add_argument("--reception-min-angle-change", type=float, default=135.0)
    parser.add_argument("--reception-window-sec", type=float, default=4 / 30, help="Seconds before/after a ball point used to estimate trajectory change.")
    parser.add_argument("--reception-window", dest="reception_window_frames", type=int, default=None, help="Deprecated frame override for --reception-window-sec.")
    parser.add_argument("--reception-min-gap-sec", type=float, default=5 / 30, help="Seconds to skip after the serve window before searching for reception.")
    parser.add_argument("--reception-min-frame-gap", dest="reception_min_gap_frames", type=int, default=None, help="Deprecated frame override for --reception-min-gap-sec.")
    parser.add_argument("--action-min-gap-sec", type=float, default=12 / 30, help="Group sharp ball changes closer than this many seconds into one action.")
    parser.add_argument("--action-min-frame-gap", dest="action_min_gap_frames", type=int, default=None, help="Deprecated frame override for --action-min-gap-sec.")
    parser.add_argument(
        "--receive-wait-prob-threshold",
        type=float,
        default=0.33,
        help="Classify an action as receive when SVM wait probability is below this threshold.",
    )
    parser.add_argument(
        "--receive-prob-threshold",
        type=float,
        default=0.33,
        help="Mark the first action with receive_top or receive_bottom probability above this threshold as reception.",
    )
    parser.add_argument("--receiver-window-sec", type=float, default=4 / 30, help="Find the nearest player within +/- this many seconds.")
    parser.add_argument("--receiver-frame-window", dest="receiver_window_frames", type=int, default=None, help="Deprecated frame override for --receiver-window-sec.")
    parser.add_argument("--receiver-max-distance", type=float, default=180.0)
    parser.add_argument("--receiver-dispute-margin", type=float, default=35.0)
    parser.add_argument(
        "--receiver-known-player-bonus",
        type=float,
        default=110.0,
        help="Pixel-equivalent preference for an identified player over an unknown track.",
    )
    parser.add_argument(
        "--receiver-prediction-penalty",
        type=float,
        default=35.0,
        help="Pixel-equivalent penalty for an unobserved/predicted player box.",
    )
    parser.add_argument("--player-draw-max-gap-sec", type=float, default=12 / 30, help="Draw the latest player box within this many seconds, matching test_track_video.py.")
    parser.add_argument("--player-draw-max-gap", dest="player_draw_max_gap_frames", type=int, default=None, help="Deprecated frame override for --player-draw-max-gap-sec.")
    parser.add_argument(
        "--pass-origin-lookback-sec",
        type=float,
        default=0.20,
        help=(
            "Seconds before the pass searched for the passer foot point with maximum image y. "
            "This uses the grounded/takeoff position when the setter makes a jump pass."
        ),
    )
    parser.add_argument(
        "--pass-origin-lookback",
        dest="pass_origin_lookback_frames",
        type=int,
        default=None,
        help="Deprecated frame override for --pass-origin-lookback-sec.",
    )
    parser.add_argument(
        "--pose-svm-model",
        default=None,
        help="Optional pose SVM joblib from labeling/train_pose_svm.py. Adds action probabilities for the receiver.",
    )
    parser.add_argument(
        "--pose-model",
        default="external/mediapipe/pose_landmarker_lite.task",
        help="MediaPipe pose landmarker model used when --pose-svm-model is enabled.",
    )
    parser.add_argument(
        "--pose-min-detection-confidence",
        type=float,
        default=0.20,
        help="MediaPipe pose confidence threshold for action SVM crops.",
    )
    parser.add_argument("--max-ball-gap-sec", type=float, default=0.0, help="Seconds of missing ball detections to fill with predicted points.")
    parser.add_argument("--max-ball-gap", dest="max_ball_gap_frames", type=int, default=None, help="Deprecated frame override for --max-ball-gap-sec.")
    parser.add_argument("--ball-max-jump", type=float, default=100.0)
    parser.add_argument("--ball-reacquire-gap-sec", type=float, default=5 / 30, help="Missing-ball seconds before using the relaxed reacquisition jump limit.")
    parser.add_argument("--ball-reacquire-gap", dest="ball_reacquire_gap_frames", type=int, default=None, help="Deprecated frame override for --ball-reacquire-gap-sec.")
    parser.add_argument("--ball-reacquire-max-jump", type=float, default=1000.0)
    parser.add_argument("--ball-reset-gap-sec", type=float, default=5 / 30, help="Missing/rejected-ball seconds before resetting tracking state.")
    parser.add_argument("--ball-reset-gap", dest="ball_reset_gap_frames", type=int, default=None, help="Deprecated frame override for --ball-reset-gap-sec.")
    parser.add_argument("--label-hold-sec", type=float, default=1.5)
    parser.add_argument("--trail-length-sec", type=float, default=45 / 30, help="Seconds of recent ball history drawn on the video.")
    parser.add_argument("--trail-length", dest="trail_length_frames", type=int, default=None, help="Deprecated frame override for --trail-length-sec.")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    deps = import_dependencies()
    cv2 = deps["cv2"]

    video_path = resolve_path(args.video)
    ball_track_path = resolve_path(args.ball_track)
    tracking_path = resolve_path(args.tracking_json)
    layout_path = resolve_path(args.layout)
    reception_zones_path = resolve_path(args.reception_zones)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    if not ball_track_path.exists():
        raise SystemExit(f"Ball track not found: {ball_track_path}")
    if not tracking_path.exists():
        raise SystemExit(f"Tracking JSON not found: {tracking_path}")
    if args.team_filter != "none" and not layout_path.exists():
        raise SystemExit(f"Layout file not found: {layout_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = args.max_frames if args.max_frames > 0 else frame_count
    try:
        temporal_parameters = resolve_temporal_parameters(args, fps)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    raw_ball_points = load_ball_track(ball_track_path)
    ball_points = filter_ball_track(
        raw_ball_points,
        max_frame=max_frames - 1,
        max_gap=args.max_ball_gap,
        max_jump=args.ball_max_jump,
        reacquire_gap=args.ball_reacquire_gap,
        reacquire_max_jump=args.ball_reacquire_max_jump,
        reset_gap=args.ball_reset_gap,
    )
    if len(ball_points) < args.serve_window + args.reception_window + 2:
        raise SystemExit("Not enough filtered ball points to classify the rally.")

    court_polygon = load_layout_polygon(layout_path) if args.team_filter != "none" else []
    reception_zones = load_reception_zones(reception_zones_path)
    all_player_boxes_by_frame = load_player_boxes(tracking_path)
    render_player_boxes_by_frame = dedupe_player_boxes_by_label(all_player_boxes_by_frame)
    player_boxes_by_frame = filter_player_boxes_by_team(
        render_player_boxes_by_frame,
        court_polygon,
        args.team_filter,
    )
    player_boxes_by_frame = dedupe_player_boxes_by_label(player_boxes_by_frame)
    render_player_tracks_by_id = player_tracks_from_boxes(render_player_boxes_by_frame)
    scoring_player_tracks_by_id = player_tracks_from_boxes(
        filter_player_boxes_by_team(all_player_boxes_by_frame, court_polygon, args.team_filter)
    )
    pose_classifier = load_pose_classifier(args) if args.pose_svm_model else None
    serve, reception = classify_serve_and_reception(ball_points, max_frames, args)
    serve["time_sec"] = serve["frame"] / fps
    if serve.get("contact_captured"):
        contact_player = find_serve_contact_player(
            cv2=cv2,
            video_path=video_path,
            player_boxes_by_frame=all_player_boxes_by_frame,
            ball_x=float(serve["x"]),
            ball_y=float(serve["y"]),
            contact_frame=int(serve["frame"]),
            serving_team=str(serve.get("serving_team") or "unknown"),
            frame_window=args.receiver_frame_window,
            max_distance=float(args.serve_contact_player_max_distance),
            pose_classifier=pose_classifier,
        )
        if contact_player is not None:
            serve["contact_player"] = contact_player
    actions = classify_actions_after_serve(
        cv2=cv2,
        video_path=video_path,
        ball_points=ball_points,
        player_boxes_by_frame=player_boxes_by_frame,
        serve=serve,
        fps=fps,
        args=args,
        pose_classifier=pose_classifier,
    )
    reception_evaluation = evaluate_reception_quality(
        actions,
        reception_zones,
        args.receive_prob_threshold,
        player_tracks_by_id=scoring_player_tracks_by_id,
        pass_origin_lookback_frames=args.pass_origin_lookback,
    )
    reception, receiver = first_receive_action(actions)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    json_path = output_dir / f"{stem}_serve_receive.json"
    reception_eval_path = output_dir / f"{stem}_reception_evaluation.json"
    annotated_path = output_dir / f"{stem}_serve_receive_annotated.mp4"

    result = {
        "video_id": stem,
        "video_path": str(video_path),
        "ball_track": str(ball_track_path),
        "tracking_json": str(tracking_path),
        "layout": str(layout_path) if args.team_filter != "none" else None,
        "reception_zones": str(reception_zones_path) if reception_zones else None,
        "player_team_filter": args.team_filter,
        "fps": fps,
        "temporal_parameters": temporal_parameters,
        "raw_ball_points": len(raw_ball_points),
        "filtered_ball_points": len(ball_points),
        "assumptions": {
            "serve_model": "contact-first compact toss followed by validated outbound flight, with smooth-flight fallback",
            "action_model": "each grouped strong trajectory reversal after serve flight",
            "receive_model": f"SVM wait probability below {args.receive_wait_prob_threshold}",
            "one_rally_per_video": True,
            "max_receptions": "unlimited",
        },
        "serve": serve,
        "actions": actions,
        "reception_evaluation": reception_evaluation,
        "reception": reception,
        "receiver": receiver,
    }
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    reception_eval_path.write_text(
        json.dumps(reception_evaluation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if not args.no_video:
        write_annotated_video(
            cv2=cv2,
            video_path=video_path,
            output_path=annotated_path,
            fps=fps,
            size=(width, height),
            max_frames=max_frames,
            ball_points=ball_points,
            player_boxes_by_frame=render_player_boxes_by_frame,
            player_tracks_by_id=render_player_tracks_by_id,
            serve=serve,
            actions=actions,
            reception_evaluation=reception_evaluation,
            reception_zones=reception_zones,
            label_hold_frames=max(1, int(round(args.label_hold_sec * fps))),
            player_draw_max_gap=args.player_draw_max_gap,
            trail_length=args.trail_length,
        )
    if pose_classifier is not None:
        pose_classifier["pose_estimator"].close()

    print(
        f"Serve: frame={serve['frame']} team={serve['serving_team']} "
        f"type={serve.get('serve_type', 'unknown')} contact_captured={serve.get('contact_captured', False)} "
        f"speed={serve['avg_speed_px_per_frame']:.2f} smooth_angle={serve['mean_angle_change_deg']:.1f}"
    )
    if not actions:
        print("Actions: none")
    else:
        print(f"Actions: {len(actions)}")
        for action in actions:
            receiver_text = "none"
            if action.get("receiver") is not None:
                receiver_text = action["receiver"].get("player_id") or str(action["receiver"].get("track_id"))
            pose_action = (action.get("receiver") or {}).get("pose_action")
            predicted_label = pose_action.get("predicted_label") if pose_action else None
            print(
                f"Action: frame={action['frame']} type={action.get('action_type')} "
                f"label={predicted_label or 'none'} "
                f"angle={action['angle_change_deg']:.1f} player={receiver_text}"
            )
            if pose_action:
                if pose_action.get("probabilities"):
                    print(f"Action probabilities: {format_probability_vector(pose_action['probabilities'])}")
                elif pose_action.get("error"):
                    print(f"Action probabilities: unavailable ({pose_action['error']})")
                    if pose_action.get("pose_attempt_errors"):
                        print(f"Pose attempts: {'; '.join(pose_action['pose_attempt_errors'])}")
    if reception is None:
        print("Reception: none")
    else:
        print(f"First receive: frame={reception['frame']} angle={reception['angle_change_deg']:.1f}")
    if receiver is not None:
        print(f"First receiver: {receiver.get('player_id') or receiver.get('track_id')} disputed={receiver['disputed']}")
        if receiver.get("pose_action"):
            pose_action = receiver["pose_action"]
            print(
                f"First receiver action: {pose_action.get('predicted_label')} "
                f"p={pose_action.get('predicted_probability', 0.0):.3f}"
            )
            if pose_action.get("probabilities"):
                print(f"First receiver action probabilities: {format_probability_vector(pose_action['probabilities'])}")
    if reception_evaluation.get("ok"):
        print(
            f"Reception evaluation: receiver={reception_evaluation.get('receiver')} "
            f"score={reception_evaluation.get('score')} reason={reception_evaluation.get('score_reason')}"
        )
        if reception_evaluation.get("pass"):
            pass_info = reception_evaluation["pass"]
            print(f"Pass: frame={pass_info.get('frame')} passer={pass_info.get('passer')}")
            origin = pass_info.get("origin_court_anchor") or {}
            if origin:
                print(
                    f"Pass origin: frame={origin.get('frame')} "
                    f"point=({origin.get('x'):.1f}, {origin.get('y'):.1f}) "
                    f"method={origin.get('method')}"
                )
    else:
        print(f"Reception evaluation: unavailable ({reception_evaluation.get('reason')})")
    print(f"Saved JSON: {json_path}")
    print(f"Saved reception evaluation: {reception_eval_path}")
    if not args.no_video:
        print(f"Saved annotated video: {annotated_path}")
    return 0


def resolve_temporal_parameters(args: argparse.Namespace, fps: float) -> dict[str, dict[str, Any]]:
    specifications = [
        ("serve_window", "serve_window_sec", "serve_window_frames", "--serve-window-sec", "--serve-window", 1),
        ("reception_window", "reception_window_sec", "reception_window_frames", "--reception-window-sec", "--reception-window", 1),
        (
            "reception_min_frame_gap",
            "reception_min_gap_sec",
            "reception_min_gap_frames",
            "--reception-min-gap-sec",
            "--reception-min-frame-gap",
            0,
        ),
        (
            "action_min_frame_gap",
            "action_min_gap_sec",
            "action_min_gap_frames",
            "--action-min-gap-sec",
            "--action-min-frame-gap",
            0,
        ),
        (
            "receiver_frame_window",
            "receiver_window_sec",
            "receiver_window_frames",
            "--receiver-window-sec",
            "--receiver-frame-window",
            0,
        ),
        (
            "player_draw_max_gap",
            "player_draw_max_gap_sec",
            "player_draw_max_gap_frames",
            "--player-draw-max-gap-sec",
            "--player-draw-max-gap",
            0,
        ),
        (
            "pass_origin_lookback",
            "pass_origin_lookback_sec",
            "pass_origin_lookback_frames",
            "--pass-origin-lookback-sec",
            "--pass-origin-lookback",
            0,
        ),
        (
            "serve_toss_search",
            "serve_toss_search_sec",
            "serve_toss_search_frames",
            "--serve-toss-search-sec",
            "--serve-toss-search",
            1,
        ),
        (
            "serve_contact_max_gap",
            "serve_contact_max_gap_sec",
            "serve_contact_max_gap_frames",
            "--serve-contact-max-gap-sec",
            "--serve-contact-max-gap",
            1,
        ),
        (
            "serve_contact_flight",
            "serve_contact_flight_sec",
            "serve_contact_flight_frames",
            "--serve-contact-flight-sec",
            "--serve-contact-flight",
            1,
        ),
        ("max_ball_gap", "max_ball_gap_sec", "max_ball_gap_frames", "--max-ball-gap-sec", "--max-ball-gap", 0),
        (
            "ball_reacquire_gap",
            "ball_reacquire_gap_sec",
            "ball_reacquire_gap_frames",
            "--ball-reacquire-gap-sec",
            "--ball-reacquire-gap",
            1,
        ),
        (
            "ball_reset_gap",
            "ball_reset_gap_sec",
            "ball_reset_gap_frames",
            "--ball-reset-gap-sec",
            "--ball-reset-gap",
            1,
        ),
        ("trail_length", "trail_length_sec", "trail_length_frames", "--trail-length-sec", "--trail-length", 1),
    ]
    resolved = {}
    for target, seconds_attr, frames_attr, seconds_option, frames_option, minimum in specifications:
        resolved[target] = resolve_temporal_option(
            args,
            fps=fps,
            target_attr=target,
            seconds_attr=seconds_attr,
            legacy_frames_attr=frames_attr,
            seconds_option=seconds_option,
            legacy_frames_option=frames_option,
            minimum_frames=minimum,
        )
    return resolved


def import_dependencies() -> dict[str, Any]:
    matplotlib_cache = ROOT / ".cache" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing package: opencv-python") from exc
    return {"cv2": cv2}


def load_pose_classifier(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import joblib  # type: ignore
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing package for pose SVM inference. Install with: ./venv/bin/python -m pip install scikit-learn joblib") from exc

    labeling_dir = ROOT / "labeling"
    if str(labeling_dir) not in sys.path:
        sys.path.insert(0, str(labeling_dir))
    try:
        import build_action_pose_dataset as pose_dataset  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Could not import pose helpers from {labeling_dir}") from exc

    model_path = resolve_path(args.pose_svm_model)
    if not model_path.exists():
        raise SystemExit(f"Pose SVM model not found: {model_path}")

    artifact = joblib.load(model_path)
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    feature_names = artifact.get("feature_names", []) if isinstance(artifact, dict) else []
    labels = artifact.get("labels", []) if isinstance(artifact, dict) else []
    if not feature_names:
        raise SystemExit(f"Pose SVM artifact has no feature_names: {model_path}")
    if not hasattr(model, "predict_proba"):
        raise SystemExit(
            f"Pose SVM model has no predict_proba: {model_path}\n"
            "Retrain with: ./venv/bin/python labeling/train_pose_svm.py --landmarks main --include-visibility"
        )

    mp = pose_dataset.import_mediapipe()
    estimator_args = argparse.Namespace(
        pose_model=args.pose_model,
        pose_min_detection_confidence=args.pose_min_detection_confidence,
    )
    pose_estimator = pose_dataset.create_pose_estimator(mp, np, estimator_args)
    return {
        "np": np,
        "model": model,
        "feature_names": feature_names,
        "labels": labels,
        "pose_dataset": pose_dataset,
        "pose_estimator": pose_estimator,
        "model_path": str(model_path),
    }


def predict_receiver_action(
    cv2: Any,
    video_path: Path,
    receiver: dict[str, Any],
    classifier: dict[str, Any],
) -> dict[str, Any]:
    pose_dataset = classifier["pose_dataset"]
    frame_idx = int(receiver["frame"])
    frame = read_video_frame(cv2, video_path, frame_idx)
    if frame is None:
        return {"ok": False, "error": f"could not read frame {frame_idx}"}

    bbox_dict = receiver["bbox"]
    bbox = (
        float(bbox_dict["x1"]),
        float(bbox_dict["y1"]),
        float(bbox_dict["x2"]),
        float(bbox_dict["y2"]),
    )
    pose_result = detect_pose_for_bbox(cv2, frame, bbox, classifier["pose_estimator"])
    if not pose_result["ok"]:
        return pose_result

    landmarks = pose_dataset.convert_landmarks(
        pose_result["pose_landmarks"],
        crop_origin=pose_result["crop_origin"],
        crop_shape=pose_result["crop_shape"],
        frame_shape=frame.shape,
        bbox=bbox,
    )
    by_name = {str(point["name"]): point for point in landmarks}
    feature_values = []
    missing = []
    for feature_name in classifier["feature_names"]:
        landmark_name, coord_name = split_pose_feature_name(feature_name)
        point = by_name.get(landmark_name)
        if point is None or point.get(coord_name) in (None, ""):
            missing.append(feature_name)
            feature_values.append(0.0)
            continue
        feature_values.append(float(point[coord_name]))

    X = classifier["np"].array([feature_values], dtype="float32")
    probabilities = classifier["model"].predict_proba(X)[0]
    model_classes = [str(item) for item in getattr(classifier["model"], "classes_", classifier["labels"])]
    probability_by_label = {
        label: float(probability)
        for label, probability in sorted(zip(model_classes, probabilities), key=lambda item: item[1], reverse=True)
    }
    predicted_label = max(probability_by_label, key=probability_by_label.get)
    return {
        "ok": True,
        "frame": frame_idx,
        "model_path": classifier["model_path"],
        "predicted_label": predicted_label,
        "predicted_probability": probability_by_label[predicted_label],
        "probabilities": probability_by_label,
        "missing_features": missing,
        "pose_attempt": pose_result["attempt"],
    }


def detect_pose_for_bbox(
    cv2: Any,
    frame: Any,
    bbox: tuple[float, float, float, float],
    pose_estimator: Any,
) -> dict[str, Any]:
    attempts = [
        {"name": "padded", "pad_x": 0.12, "pad_y": 0.08, "square": False, "min_size": 0},
        {"name": "wide_padded", "pad_x": 0.35, "pad_y": 0.25, "square": False, "min_size": 0},
        {"name": "square", "pad_x": 0.30, "pad_y": 0.30, "square": True, "min_size": 0},
        {"name": "square_upscaled", "pad_x": 0.45, "pad_y": 0.35, "square": True, "min_size": 384},
    ]
    errors = []
    for attempt in attempts:
        crop, crop_origin, crop_shape = crop_bbox_for_pose(
            frame=frame,
            bbox=bbox,
            pad_x_ratio=float(attempt["pad_x"]),
            pad_y_ratio=float(attempt["pad_y"]),
            square=bool(attempt["square"]),
        )
        if crop.size == 0:
            errors.append(f"{attempt['name']}: empty crop")
            continue
        input_crop = resize_min_side(cv2, crop, int(attempt["min_size"]))
        rgb = cv2.cvtColor(input_crop, cv2.COLOR_BGR2RGB)
        pose_landmarks = pose_estimator.process(rgb)
        if pose_landmarks:
            return {
                "ok": True,
                "pose_landmarks": pose_landmarks,
                "crop_origin": crop_origin,
                "crop_shape": crop_shape,
                "attempt": attempt["name"],
            }
        errors.append(f"{attempt['name']}: no pose landmarks")
    return {"ok": False, "error": "no pose landmarks for receiver", "pose_attempt_errors": errors}


def crop_bbox_for_pose(
    frame: Any,
    bbox: tuple[float, float, float, float],
    pad_x_ratio: float,
    pad_y_ratio: float,
    square: bool,
) -> tuple[Any, tuple[int, int], tuple[int, int, int]]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if square:
        side = max(box_w * (1.0 + pad_x_ratio * 2.0), box_h * (1.0 + pad_y_ratio * 2.0))
        nx1 = cx - side / 2
        nx2 = cx + side / 2
        ny1 = cy - side / 2
        ny2 = cy + side / 2
    else:
        nx1 = x1 - box_w * pad_x_ratio
        nx2 = x2 + box_w * pad_x_ratio
        ny1 = y1 - box_h * pad_y_ratio
        ny2 = y2 + box_h * pad_y_ratio
    ix1 = max(0, int(round(nx1)))
    iy1 = max(0, int(round(ny1)))
    ix2 = min(w, int(round(nx2)))
    iy2 = min(h, int(round(ny2)))
    crop = frame[iy1:iy2, ix1:ix2]
    return crop, (ix1, iy1), crop.shape


def resize_min_side(cv2: Any, crop: Any, min_size: int) -> Any:
    if min_size <= 0:
        return crop
    h, w = crop.shape[:2]
    short_side = min(h, w)
    if short_side >= min_size:
        return crop
    scale = min_size / max(1, short_side)
    return cv2.resize(crop, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_CUBIC)


def split_pose_feature_name(feature_name: str) -> tuple[str, str]:
    for suffix in ("_bbox_x", "_bbox_y", "_visibility", "_z", "_x", "_y"):
        if feature_name.endswith(suffix):
            coord = suffix[1:]
            return feature_name[: -len(suffix)], coord
    raise ValueError(f"Unsupported pose feature name: {feature_name}")


def read_video_frame(cv2: Any, video_path: Path, frame_idx: int) -> Any | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_ball_track(path: Path) -> list[BallPoint]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("ball_track", data.get("ball_positions", data if isinstance(data, list) else []))
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    points = []
    for row in rows:
        x = row.get("x", row.get("X"))
        y = row.get("y", row.get("Y"))
        if x in (None, "", "-1") or y in (None, "", "-1"):
            continue
        width = float(row.get("width", row.get("W", 10)) or 10)
        height = float(row.get("height", row.get("H", width)) or width)
        points.append(
            BallPoint(
                frame=int(float(row.get("frame", row.get("Frame", 0)) or 0)),
                x=float(x),
                y=float(y),
                radius=max(5.0, (width + height) / 4),
                confidence=float(row.get("confidence", row.get("Confidence", 1.0)) or 0),
                tracklet_id=parse_optional_int(row.get("tracklet_id", row.get("ball_tracklet_id"))),
            )
        )
    return sorted(points, key=lambda item: item.frame)


def parse_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def filter_ball_track(
    points: list[BallPoint],
    max_frame: int,
    max_gap: int,
    max_jump: float,
    reacquire_gap: int,
    reacquire_max_jump: float,
    reset_gap: int,
) -> list[BallPoint]:
    by_frame = {point.frame: point for point in points}
    if not by_frame:
        return []
    state = BallState()
    filtered = []
    for frame in range(min(by_frame), min(max_frame, max(by_frame)) + 1):
        accepted = update_ball_state(
            point=by_frame.get(frame),
            state=state,
            frame=frame,
            max_gap=max_gap,
            max_jump=max_jump,
            reacquire_gap=reacquire_gap,
            reacquire_max_jump=reacquire_max_jump,
            reset_gap=reset_gap,
        )
        if accepted is not None:
            filtered.append(accepted)
    return filtered


def update_ball_state(
    point: BallPoint | None,
    state: BallState,
    frame: int,
    max_gap: int,
    max_jump: float,
    reacquire_gap: int,
    reacquire_max_jump: float,
    reset_gap: int,
) -> BallPoint | None:
    if state.missed >= max(1, reset_gap):
        state.last = None
        state.vx = 0.0
        state.vy = 0.0
        state.missed = 0

    predicted = state.predict(frame)
    effective_max_jump = reacquire_max_jump if state.missed >= max(1, reacquire_gap) else max_jump
    if point is not None and is_valid_ball_jump(point, predicted, state.last is not None, effective_max_jump):
        state.update(point)
        return point
    state.missed += 1
    if predicted is not None and state.missed <= max_gap:
        state.last = predicted
        return predicted
    return None


def is_valid_ball_jump(point: BallPoint, predicted: BallPoint | None, has_track: bool, max_jump: float) -> bool:
    if not has_track or predicted is None:
        return point.confidence >= 0.34
    dt = max(1, point.frame - predicted.frame)
    return math.hypot(point.x - predicted.x, point.y - predicted.y) <= max_jump * dt


def load_player_boxes(path: Path) -> dict[int, list[PlayerBox]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    visible = data.get("visible_player_detections") or []
    if visible:
        return load_visible_player_boxes(visible)

    by_frame: dict[int, list[PlayerBox]] = {}
    for track in data.get("player_tracks", []):
        track_id = str(track.get("track_id"))
        for det in track.get("detections", []):
            bbox = det.get("bbox") or {}
            frame = int(det.get("frame", 0))
            box = PlayerBox(
                frame=frame,
                track_id=track_id,
                player_id=det.get("player_id"),
                bbox=(float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])),
                confidence=det.get("confidence"),
                team=str(det.get("team") or track.get("team") or "unknown"),
                observed=bool(det.get("observed", True)),
            )
            by_frame.setdefault(frame, []).append(box)
    return by_frame


def load_visible_player_boxes(rows: list[dict[str, Any]]) -> dict[int, list[PlayerBox]]:
    by_frame: dict[int, list[PlayerBox]] = {}
    for row in rows:
        frame = int(row.get("frame", 0))
        for det in row.get("detections", []):
            bbox = det.get("bbox") or {}
            box = PlayerBox(
                frame=frame,
                track_id=str(det.get("track_id")),
                player_id=det.get("player_id") or det.get("raw_player_id"),
                bbox=(float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])),
                confidence=det.get("confidence"),
                team=str(det.get("team") or "unknown"),
                observed=bool(det.get("observed", True)),
            )
            by_frame.setdefault(frame, []).append(box)
    return by_frame


def load_layout_polygon(path: Path) -> list[tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    points = data.get("near_side_boundary_px") or []
    return [(float(point["x"]), float(point["y"])) for point in points if "x" in point and "y" in point]


def load_reception_zones(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    zones = []
    for zone in data.get("zones", []):
        points = zone.get("polygon_px") or []
        polygon = [(float(point["x"]), float(point["y"])) for point in points if "x" in point and "y" in point]
        if len(polygon) < 3:
            continue
        zones.append(
            {
                "zone_id": str(zone.get("zone_id") or f"score_{zone.get('score', 0)}"),
                "label": str(zone.get("label") or ""),
                "score": float(zone.get("score", 0)),
                "polygon": polygon,
            }
        )
    return zones


def filter_player_boxes_by_team(
    boxes_by_frame: dict[int, list[PlayerBox]],
    polygon: list[tuple[float, float]],
    mode: str,
) -> dict[int, list[PlayerBox]]:
    if mode == "none":
        return boxes_by_frame
    if mode == "classified-near":
        return {
            frame: [box for box in boxes if box.team == "near"]
            for frame, boxes in boxes_by_frame.items()
        }
    if not polygon:
        return boxes_by_frame
    filtered = {}
    for frame, boxes in boxes_by_frame.items():
        court_hits = [box for box in boxes if box_intersects_polygon(box.bbox, polygon)]
        if mode == "court-intersection":
            filtered[frame] = court_hits
            continue
        if mode == "court-nearest-6":
            front_a, front_b = polygon[0], polygon[1] if len(polygon) > 1 else polygon[0]
            court_min_y = min(point[1] for point in polygon)

            def box_tier(box: PlayerBox) -> int:
                if box in court_hits:
                    return 0
                if box_foot(box.bbox)[1] >= court_min_y:
                    return 1
                return 2

            ranked = sorted(
                boxes,
                key=lambda box: (
                    box_tier(box),
                    distance_point_to_segment(box_foot(box.bbox), front_a, front_b),
                    -(box.confidence or 0.0),
                ),
            )
            filtered[frame] = ranked[:6]
            continue
        filtered[frame] = boxes
    return filtered


def dedupe_player_boxes_by_label(boxes_by_frame: dict[int, list[PlayerBox]]) -> dict[int, list[PlayerBox]]:
    deduped = {}
    for frame, boxes in boxes_by_frame.items():
        best_by_label: dict[str, PlayerBox] = {}
        unlabeled = []
        for box in boxes:
            if not box.player_id:
                unlabeled.append(box)
                continue
            current = best_by_label.get(box.player_id)
            if current is None or player_box_rank(box) > player_box_rank(current):
                best_by_label[box.player_id] = box
        deduped[frame] = [*best_by_label.values(), *unlabeled]
    return deduped


def player_tracks_from_boxes(boxes_by_frame: dict[int, list[PlayerBox]]) -> dict[str, list[PlayerBox]]:
    tracks: dict[str, list[PlayerBox]] = {}
    for boxes in boxes_by_frame.values():
        for box in boxes:
            tracks.setdefault(box.track_id, []).append(box)
    for boxes in tracks.values():
        boxes.sort(key=lambda box: box.frame)
    return tracks


def latest_player_boxes_near_frame(
    tracks_by_id: dict[str, list[PlayerBox]],
    frame: int,
    max_gap: int,
) -> list[PlayerBox]:
    boxes = []
    for track_boxes in tracks_by_id.values():
        best = None
        for box in reversed(track_boxes):
            if box.frame > frame:
                continue
            if frame - box.frame <= max_gap:
                best = box
            break
        if best is not None:
            boxes.append(best)
    return dedupe_player_boxes_by_label({frame: boxes}).get(frame, [])


def player_box_rank(box: PlayerBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box.bbox
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return (box.confidence or 0.0, area)


def classify_serve_and_reception(
    points: list[BallPoint],
    frame_count: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    smoothed = smooth_points(points)
    reception_candidates = detect_reception_candidates(smoothed, args)

    # A tossed serve and a reception both create a sharp trajectory change.
    # Resolve a compact toss -> contact -> outbound-flight transition first;
    # otherwise the serve contact itself is unconditionally consumed as the
    # first reception candidate.
    contact_serve = detect_toss_contact_serve(
        points=smoothed,
        reception_candidates=reception_candidates,
        frame_count=frame_count,
        args=args,
    )
    if contact_serve is not None:
        reception = first_reception_after(smoothed, int(contact_serve["window_end_frame"]), args)
        return contact_serve, reception

    if reception_candidates:
        reception = reception_candidates[0]
        serve = detect_smooth_serve_before(
            points=smoothed,
            frame_count=frame_count,
            before_frame=reception["frame"],
            args=args,
        )
        if serve is not None:
            serve["serving_team"] = "far"
            serve["receiving_team"] = "near"
            serve["detection_reason"] = "smooth early serve flight before strong trajectory reversal"
            return serve, reception

        # Do not search progressively later contact candidates: a rally flight
        # can otherwise be relabeled as a serve merely because the true serve
        # contact/flight was not captured. Keep the first contact candidate and
        # record the earliest retained ball point as an uncaptured-flight hint.
        return uncaptured_serve_fallback(smoothed, reception), reception

    serve = detect_smooth_serve_before(
        points=smoothed,
        frame_count=frame_count,
        before_frame=None,
        args=args,
    )
    if serve is None:
        raise SystemExit("Could not find an early smooth serve-like ball flight.")

    serve["detection_reason"] = "early smooth serve flight; no later reception reversal found"
    if serve["y_speed_px_per_frame"] > 0:
        serve["serving_team"] = "far"
        serve["receiving_team"] = "near"
        reception = first_reception_after(smoothed, serve["window_end_frame"], args)
        return serve, reception

    serve["serving_team"] = "near"
    serve["receiving_team"] = "far"
    return serve, None


def detect_toss_contact_serve(
    points: list[BallPoint],
    reception_candidates: list[dict[str, Any]],
    frame_count: int,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Find an early toss/contact before interpreting sharp turns as receptions.

    A contact candidate must end a compact, predominantly vertical toss and be
    followed by a smooth outbound trajectory.  The outbound validation is
    deliberately longer than the legacy serve window for rear-camera views,
    but is cut off immediately before the next sharp turn so a real reception
    remains available to the action classifier.
    """

    if not points or not reception_candidates:
        return None

    ordered = sorted(points, key=lambda point: point.frame)
    by_frame = {point.frame: point for point in ordered}
    search_until = int(frame_count * float(args.serve_search_ratio))
    validation_frames = int(getattr(args, "serve_contact_flight", args.serve_window))
    min_flight_speed = float(getattr(args, "serve_contact_min_speed", args.serve_min_speed))
    transitions: list[tuple[float, dict[str, Any]]] = []

    for index, candidate in enumerate(reception_candidates):
        contact_frame = int(candidate["frame"])
        if contact_frame >= search_until:
            continue
        contact = by_frame.get(contact_frame)
        if contact is None:
            contact = nearest_point(by_frame, contact_frame, -1, max(1, args.reception_window))
        if contact is None:
            continue

        toss_points = compact_toss_suffix(
            ordered,
            end_frame=contact.frame,
            max_frames=int(args.serve_toss_search),
            max_x_drift=float(args.serve_toss_max_x_drift),
        )
        if len(toss_points) < 3:
            continue
        toss = serve_toss_stats(toss_points)
        if toss["x_span_px"] > float(args.serve_toss_max_x_drift):
            continue
        if toss["vertical_motion_px"] < float(args.serve_toss_min_vertical_motion):
            continue
        if toss["vertical_to_horizontal_motion_ratio"] < float(args.serve_toss_min_vertical_x_ratio):
            continue

        next_contact_frame = None
        for later in reception_candidates[index + 1 :]:
            if int(later["frame"]) > contact.frame:
                next_contact_frame = int(later["frame"])
                break
        validation_end = contact.frame + validation_frames
        if next_contact_frame is not None:
            validation_end = min(validation_end, next_contact_frame - 1)
        if validation_end <= contact.frame:
            continue

        flight = segment_points(
            by_frame,
            contact.frame,
            validation_end - contact.frame,
            max_gap=2,
        )
        minimum_points = max(5, min(int(args.serve_window), validation_end - contact.frame + 1) // 2)
        if len(flight) < minimum_points or flight[-1].frame <= contact.frame:
            continue
        flight_stats = trajectory_stats(flight)
        if flight_stats["distance"] < float(args.serve_min_distance):
            continue
        if flight_stats["avg_speed"] < min_flight_speed:
            continue
        if flight_stats["mean_angle_change"] > float(args.serve_max_mean_angle_change):
            continue

        toss_center_x = sorted(point.x for point in toss_points)[len(toss_points) // 2]
        departure_x = max(abs(point.x - toss_center_x) for point in flight[1:])
        minimum_departure = max(10.0, toss["x_span_px"] * 1.5)
        if departure_x <= minimum_departure:
            continue

        serve_type, type_reason = classify_serve_type(toss, flight_stats, args)
        serving_team = "far" if flight_stats["y_speed"] > 0 else "near"
        receiving_team = "near" if serving_team == "far" else "far"
        score = (
            float(candidate.get("angle_change_deg", 0.0))
            + toss["vertical_motion_px"] * 0.20
            + flight_stats["distance"] * 0.10
            + flight_stats["avg_speed"]
            - contact.frame / max(1, search_until) * 10.0
        )
        transitions.append(
            (
                score,
                {
                    "frame": contact.frame,
                    "time_sec": None,
                    "x": contact.x,
                    "y": contact.y,
                    "window_end_frame": flight[-1].frame,
                    "window_end_x": flight[-1].x,
                    "window_end_y": flight[-1].y,
                    "direction": "toward_near_team" if serving_team == "far" else "toward_far_team",
                    "serving_team": serving_team,
                    "receiving_team": receiving_team,
                    "y_speed_px_per_frame": flight_stats["y_speed"],
                    "avg_speed_px_per_frame": flight_stats["avg_speed"],
                    "total_distance_px": flight_stats["distance"],
                    "mean_angle_change_deg": flight_stats["mean_angle_change"],
                    "serve_type": serve_type,
                    "serve_type_reason": type_reason,
                    "contact_captured": True,
                    "contact_detection_reason": "compact toss ending at a sharp turn followed by outbound flight",
                    "flight_start_frame": flight[1].frame,
                    "flight_start_x": flight[1].x,
                    "flight_start_y": flight[1].y,
                    "contact_to_flight_gap_frames": flight[1].frame - contact.frame,
                    "toss": toss,
                    "contact_candidate": {
                        "angle_change_deg": candidate.get("angle_change_deg"),
                        "speed_before_px_per_frame": candidate.get("speed_before_px_per_frame"),
                        "speed_after_px_per_frame": candidate.get("speed_after_px_per_frame"),
                        "outbound_validation_end_frame": flight[-1].frame,
                        "outbound_departure_x_px": departure_x,
                    },
                    "detection_reason": "contact-first toss-to-flight transition before reception classification",
                },
            )
        )

    if not transitions:
        return None

    # Prefer the earliest credible transition.  Later rally contacts can have a
    # larger raw score, but cannot be the serve once an earlier toss/contact is
    # already supported by a valid outbound flight.
    earliest_frame = min(int(item[1]["frame"]) for item in transitions)
    earliest = [item for item in transitions if int(item[1]["frame"]) == earliest_frame]
    return max(earliest, key=lambda item: item[0])[1]


def uncaptured_serve_fallback(
    points: list[BallPoint],
    first_reception: dict[str, Any],
) -> dict[str, Any]:
    before_reception = [point for point in points if point.frame < int(first_reception["frame"])]
    hint = before_reception[0] if before_reception else points[0]
    return {
        "frame": hint.frame,
        "time_sec": None,
        "x": hint.x,
        "y": hint.y,
        "window_end_frame": hint.frame,
        "window_end_x": hint.x,
        "window_end_y": hint.y,
        "direction": "unknown",
        "serving_team": "far",
        "receiving_team": "near",
        "y_speed_px_per_frame": 0.0,
        "avg_speed_px_per_frame": 0.0,
        "total_distance_px": 0.0,
        "mean_angle_change_deg": 0.0,
        "serve_type": "unknown",
        "contact_captured": False,
        "contact_detection_reason": "no valid serve flight before the first contact candidate",
        "flight_start_frame": hint.frame,
        "flight_start_x": hint.x,
        "flight_start_y": hint.y,
        "toss": None,
        "detection_reason": "serve moment not captured; first retained action is a reception candidate",
    }


def classify_actions_after_serve(
    cv2: Any,
    video_path: Path,
    ball_points: list[BallPoint],
    player_boxes_by_frame: dict[int, list[PlayerBox]],
    serve: dict[str, Any],
    fps: float,
    args: argparse.Namespace,
    pose_classifier: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    smoothed = smooth_points(ball_points)
    drawn_ball_by_frame = {point.frame: point for point in ball_points}
    min_frame = int(serve["window_end_frame"]) + args.reception_min_frame_gap
    candidates = [
        candidate
        for candidate in detect_reception_candidates(smoothed, args)
        if int(candidate["frame"]) > min_frame
    ]
    actions = []
    for index, candidate in enumerate(candidates, start=1):
        action = dict(candidate)
        snap_action_to_drawn_ball(action, drawn_ball_by_frame, args.receiver_frame_window)
        action["action_index"] = index
        action["time_sec"] = float(action.get("receiver_frame_float", action["frame"])) / fps
        action["receiver"] = find_receiver(
            player_boxes_by_frame=player_boxes_by_frame,
            ball_x=float(action["x"]),
            ball_y=float(action["y"]),
            reception_frame=float(action.get("receiver_frame_float", action["frame"])),
            frame_window=args.receiver_frame_window,
            max_distance=args.receiver_max_distance,
            dispute_margin=args.receiver_dispute_margin,
            known_player_bonus=args.receiver_known_player_bonus,
            prediction_penalty=args.receiver_prediction_penalty,
        )
        if action["receiver"] is not None and pose_classifier is not None:
            action["receiver"]["pose_action"] = predict_receiver_action(
                cv2=cv2,
                video_path=video_path,
                receiver=action["receiver"],
                classifier=pose_classifier,
            )
        action["action_type"] = classify_action_type(action, args.receive_wait_prob_threshold)
        action["is_receive"] = action["action_type"] not in {"wait", "unknown"}
        actions.append(action)
    return actions


def snap_action_to_drawn_ball(
    action: dict[str, Any],
    drawn_ball_by_frame: dict[int, BallPoint],
    max_gap: int,
) -> None:
    point = nearest_point(drawn_ball_by_frame, int(action["frame"]), 1, max_gap)
    if point is None or point.frame != int(action["frame"]):
        point = nearest_point(drawn_ball_by_frame, int(action["frame"]), -1, max_gap)
    if point is None:
        return
    action["drawn_ball_frame"] = point.frame
    action["drawn_ball_x"] = point.x
    action["drawn_ball_y"] = point.y
    action["x"] = point.x
    action["y"] = point.y


def classify_action_type(action: dict[str, Any], wait_probability_threshold: float) -> str:
    pose_action = (action.get("receiver") or {}).get("pose_action") or {}
    probabilities = pose_action.get("probabilities") or {}
    if not probabilities:
        return "unknown"
    wait_probability = float(probabilities.get("wait", 0.0))
    if wait_probability < wait_probability_threshold:
        predicted_label = str(pose_action.get("predicted_label") or "")
        return predicted_label if predicted_label and predicted_label != "wait" else "receive"
    return "wait"


def evaluate_reception_quality(
    actions: list[dict[str, Any]],
    zones: list[dict[str, Any]],
    receive_probability_threshold: float,
    player_tracks_by_id: dict[str, list[PlayerBox]] | None = None,
    pass_origin_lookback_frames: int = 0,
) -> dict[str, Any]:
    reception_action = first_action_with_receive_probability(actions, receive_probability_threshold)
    if reception_action is None:
        return {
            "ok": False,
            "reason": "no action with receive_top or receive_bottom above threshold",
            "receive_probability_threshold": receive_probability_threshold,
            "score": None,
        }

    reception_action["role"] = "reception"
    reception_action["action_type"] = "reception"
    reception_action["is_receive"] = True

    pass_action = first_pass_action_after(actions, int(reception_action["frame"]))
    receiver_id = player_label_from_action(reception_action)
    result: dict[str, Any] = {
        "ok": True,
        "receive_probability_threshold": receive_probability_threshold,
        "receiver": receiver_id,
        "reception_frame": reception_action["frame"],
        "reception_probabilities": receive_probabilities(reception_action),
        "pass": None,
        "score": -1,
        "score_reason": "no pass action after reception",
    }
    if pass_action is None:
        reception_action["reception_score"] = -1
        return result

    pass_action["role"] = "pass"
    pass_action["action_type"] = "pass"
    passer = pass_action.get("receiver") or {}
    pass_frame = int(pass_action["frame"])
    pass_time_position = find_pass_origin_anchor(
        passer=passer,
        player_tracks_by_id=player_tracks_by_id or {},
        pass_frame=pass_frame,
        lookback_frames=0,
    )
    if pass_time_position is not None:
        pass_time_anchor = (float(pass_time_position["x"]), float(pass_time_position["y"]))
    else:
        pass_time_anchor = court_anchor_from_receiver(passer)
    pass_origin = find_pass_origin_anchor(
        passer=passer,
        player_tracks_by_id=player_tracks_by_id or {},
        pass_frame=pass_frame,
        lookback_frames=pass_origin_lookback_frames,
    )
    if pass_origin is None and pass_time_anchor is not None:
        pass_origin = {
            "x": pass_time_anchor[0],
            "y": pass_time_anchor[1],
            "frame": pass_frame,
            "track_id": passer.get("track_id"),
            "player_id": passer.get("player_id"),
            "observed": None,
            "method": "pass_receiver_box_fallback",
        }
    passer_anchor = (float(pass_origin["x"]), float(pass_origin["y"])) if pass_origin else None
    pass_info = {
        "frame": pass_frame,
        "passer": player_label_from_action(pass_action),
        "vy_before_px_per_frame": pass_action.get("vy_before_px_per_frame"),
        "vy_after_px_per_frame": pass_action.get("vy_after_px_per_frame"),
        "pass_time_court_anchor": (
            {"x": pass_time_anchor[0], "y": pass_time_anchor[1]}
            if pass_time_anchor
            else None
        ),
        "origin_search_start_frame": max(0, pass_frame - max(0, pass_origin_lookback_frames)),
        "origin_search_end_frame": pass_frame,
        "origin_lookback_frames": max(0, pass_origin_lookback_frames),
        "origin_court_anchor": pass_origin,
        # Backward-compatible field: this is now the grounded/takeoff point used for scoring.
        "court_anchor": {"x": passer_anchor[0], "y": passer_anchor[1]} if passer_anchor else None,
    }
    result["pass"] = pass_info
    if passer_anchor is None:
        result["score"] = 0
        result["score_reason"] = "pass detected but passer box was unavailable"
        reception_action["reception_score"] = 0
        return result

    zone = best_reception_zone_for_point(passer_anchor, zones)
    if zone is None:
        result["score"] = 0
        result["score_reason"] = "pass detected outside marked scoring zones"
        reception_action["reception_score"] = 0
        return result

    score = float(zone["score"])
    result["score"] = score
    result["score_reason"] = f"passer inside {zone['zone_id']}"
    result["scoring_zone"] = {
        "zone_id": zone["zone_id"],
        "label": zone.get("label"),
        "score": score,
    }
    reception_action["reception_score"] = score
    return result


def find_pass_origin_anchor(
    passer: dict[str, Any],
    player_tracks_by_id: dict[str, list[PlayerBox]],
    pass_frame: int,
    lookback_frames: int,
) -> dict[str, Any] | None:
    """Find the passer's lowest foot point shortly before a pass.

    Image y grows downward, so maximum y represents the lowest/most grounded
    tracked position. This avoids scoring a jumping setter at their airborne
    pass-time position.
    """

    start_frame = max(0, int(pass_frame) - max(0, int(lookback_frames)))
    end_frame = int(pass_frame)
    track_id = str(passer.get("track_id") or "")
    player_id = str(passer.get("player_id") or "")

    candidates = [
        box
        for box in player_tracks_by_id.get(track_id, [])
        if start_frame <= box.frame <= end_frame
    ]
    method = "max_y_same_track_pre_pass"
    if not candidates and player_id:
        candidates = [
            box
            for boxes in player_tracks_by_id.values()
            for box in boxes
            if box.player_id == player_id and start_frame <= box.frame <= end_frame
        ]
        method = "max_y_same_player_pre_pass"
    if not candidates:
        return None

    observed = [box for box in candidates if box.observed]
    if observed:
        candidates = observed
    else:
        method += "_predicted"
    selected = max(
        candidates,
        key=lambda box: (box.court_anchor[1], box.frame, box.confidence or 0.0),
    )
    x, y = selected.court_anchor
    return {
        "x": x,
        "y": y,
        "frame": selected.frame,
        "track_id": selected.track_id,
        "player_id": selected.player_id,
        "observed": selected.observed,
        "method": method,
    }


def first_action_with_receive_probability(
    actions: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any] | None:
    for action in actions:
        probabilities = receive_probabilities(action)
        if max(probabilities.values(), default=0.0) >= threshold:
            return action
    return None


def receive_probabilities(action: dict[str, Any]) -> dict[str, float]:
    probabilities = ((action.get("receiver") or {}).get("pose_action") or {}).get("probabilities") or {}
    return {
        "receive_top": float(probabilities.get("receive_top", probabilities.get("recive_top", 0.0)) or 0.0),
        "receive_bottom": float(probabilities.get("receive_bottom", probabilities.get("recive_bottom", 0.0)) or 0.0),
    }


def first_pass_action_after(actions: list[dict[str, Any]], reception_frame: int) -> dict[str, Any] | None:
    for action in actions:
        if int(action["frame"]) <= reception_frame:
            continue
        vy_before = action.get("vy_before_px_per_frame")
        vy_after = action.get("vy_after_px_per_frame")
        if vy_before is None or vy_after is None:
            continue
        if float(vy_before) > 0 and float(vy_after) < 0:
            return action
    return None


def player_label_from_action(action: dict[str, Any]) -> str | None:
    receiver = action.get("receiver") or {}
    label = receiver.get("player_id") or receiver.get("track_id")
    return str(label) if label is not None else None


def court_anchor_from_receiver(receiver: dict[str, Any]) -> tuple[float, float] | None:
    bbox = receiver.get("bbox") or {}
    try:
        x1 = float(bbox["x1"])
        x2 = float(bbox["x2"])
        y2 = float(bbox["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    return ((x1 + x2) / 2, y2)


def best_reception_zone_for_point(
    point: tuple[float, float],
    zones: list[dict[str, Any]],
) -> dict[str, Any] | None:
    matches = [zone for zone in zones if point_in_polygon(point, zone["polygon"])]
    if not matches:
        return None
    return max(matches, key=lambda zone: float(zone["score"]))


def first_receive_action(actions: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for action in actions:
        action_type = str(action.get("action_type") or "")
        if action_type and action_type not in {"wait", "unknown"}:
            reception = {key: value for key, value in action.items() if key != "receiver"}
            return reception, action.get("receiver")
    return None, None


def detect_smooth_serve_before(
    points: list[BallPoint],
    frame_count: int,
    before_frame: int | None,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    by_frame = {point.frame: point for point in points}
    search_until = int(frame_count * args.serve_search_ratio)
    if before_frame is not None:
        search_until = min(search_until, before_frame - args.reception_min_frame_gap)

    candidates = []
    for start in points:
        if start.frame >= search_until:
            continue
        segment = segment_points(by_frame, start.frame, args.serve_window, max_gap=2)
        if len(segment) < max(5, args.serve_window // 2):
            continue
        end = segment[-1]
        if end.frame > search_until:
            continue
        stats = trajectory_stats(segment)
        if stats["distance"] < args.serve_min_distance:
            continue
        if stats["avg_speed"] < args.serve_min_speed:
            continue
        if stats["mean_angle_change"] > args.serve_max_mean_angle_change:
            continue

        early_score = 1.0 - min(1.0, start.frame / max(1, search_until))
        score = (
            stats["avg_speed"]
            + stats["distance"] / max(1, args.serve_window) * 0.35
            + (args.serve_max_mean_angle_change - stats["mean_angle_change"]) * 0.20
            + early_score * args.serve_early_bonus * 100
        )
        candidates.append((score, start, end, stats))

    if not candidates:
        return None

    _score, start, end, stats = max(candidates, key=lambda item: item[0])
    serve = {
        "frame": start.frame,
        "time_sec": None,
        "x": start.x,
        "y": start.y,
        "window_end_frame": end.frame,
        "window_end_x": end.x,
        "window_end_y": end.y,
        "direction": "toward_near_team" if stats["y_speed"] > 0 else "toward_far_team",
        "serving_team": "unknown",
        "receiving_team": "unknown",
        "y_speed_px_per_frame": stats["y_speed"],
        "avg_speed_px_per_frame": stats["avg_speed"],
        "total_distance_px": stats["distance"],
        "mean_angle_change_deg": stats["mean_angle_change"],
    }
    return refine_serve_contact_and_type(points, serve, args)


def refine_serve_contact_and_type(
    points: list[BallPoint],
    serve: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Separate a compact vertical toss from the outbound serve flight.

    The last point in the compact toss is treated as the contact location. A
    large visibility gap is permitted because fast serves are frequently missed
    for several frames immediately after contact. If no compact toss-to-flight
    transition is visible, the existing flight start is retained and explicitly
    marked as a fallback rather than claiming that contact was observed.
    """

    ordered = sorted(points, key=lambda point: point.frame)
    search_end = int(serve["window_end_frame"])
    search_start = max(0, int(serve["frame"]) - int(args.serve_toss_search))
    local = [point for point in ordered if search_start <= point.frame <= search_end]
    transitions: list[tuple[float, dict[str, Any]]] = []

    for previous, outbound_start in zip(local, local[1:]):
        gap = outbound_start.frame - previous.frame
        if gap <= 0 or gap > int(args.serve_contact_max_gap):
            continue

        toss_points = compact_toss_suffix(
            local,
            end_frame=previous.frame,
            max_frames=int(args.serve_toss_search),
            max_x_drift=float(args.serve_toss_max_x_drift),
        )
        if len(toss_points) < 3:
            continue

        toss = serve_toss_stats(toss_points)
        if toss["x_span_px"] > float(args.serve_toss_max_x_drift):
            continue
        if toss["vertical_motion_px"] < float(args.serve_toss_min_vertical_motion):
            continue
        if toss["vertical_to_horizontal_motion_ratio"] < float(args.serve_toss_min_vertical_x_ratio):
            continue

        by_frame = {point.frame: point for point in ordered}
        flight = [previous]
        flight.extend(
            point
            for point in segment_points(
                by_frame,
                outbound_start.frame,
                int(args.serve_window),
                max_gap=2,
            )
            if point.frame > previous.frame
        )
        if len(flight) < max(5, int(args.serve_window) // 2):
            continue
        flight_stats = trajectory_stats(flight)
        if flight_stats["distance"] < float(args.serve_min_distance):
            continue
        if flight_stats["avg_speed"] < float(args.serve_min_speed):
            continue
        if flight_stats["mean_angle_change"] > float(args.serve_max_mean_angle_change):
            continue

        toss_center_x = sorted(point.x for point in toss_points)[len(toss_points) // 2]
        departure_x = max(abs(point.x - toss_center_x) for point in flight[1:])
        # A serve must visibly leave the narrow toss column. This prevents a
        # high vertical toss itself from being selected as the serve flight.
        if departure_x <= max(10.0, toss["x_span_px"] * 1.5):
            continue

        launch_speed = math.hypot(
            outbound_start.x - previous.x,
            outbound_start.y - previous.y,
        ) / gap
        score = (
            flight_stats["avg_speed"]
            + flight_stats["distance"] / max(1, int(args.serve_window))
            + launch_speed
            + toss["vertical_motion_px"] * 0.10
        )
        transitions.append(
            (
                score,
                {
                    "contact": previous,
                    "outbound_start": outbound_start,
                    "flight_end": flight[-1],
                    "flight_stats": flight_stats,
                    "toss": toss,
                    "launch_speed": launch_speed,
                },
            )
        )

    if not transitions:
        serve.update(
            {
                "serve_type": "unknown",
                "contact_captured": False,
                "contact_detection_reason": "no compact vertical toss-to-flight transition was visible",
                "flight_start_frame": int(serve["frame"]),
                "flight_start_x": float(serve["x"]),
                "flight_start_y": float(serve["y"]),
                "toss": None,
            }
        )
        return serve

    _score, transition = max(transitions, key=lambda item: item[0])
    contact: BallPoint = transition["contact"]
    outbound_start: BallPoint = transition["outbound_start"]
    flight_end: BallPoint = transition["flight_end"]
    flight_stats = transition["flight_stats"]
    toss = transition["toss"]

    serve_type, type_reason = classify_serve_type(toss, flight_stats, args)

    serve.update(
        {
            "frame": contact.frame,
            "x": contact.x,
            "y": contact.y,
            "window_end_frame": flight_end.frame,
            "window_end_x": flight_end.x,
            "window_end_y": flight_end.y,
            "direction": "toward_near_team" if flight_stats["y_speed"] > 0 else "toward_far_team",
            "y_speed_px_per_frame": flight_stats["y_speed"],
            "avg_speed_px_per_frame": flight_stats["avg_speed"],
            "total_distance_px": flight_stats["distance"],
            "mean_angle_change_deg": flight_stats["mean_angle_change"],
            "serve_type": serve_type,
            "serve_type_reason": type_reason,
            "contact_captured": True,
            "contact_detection_reason": "last compact toss point before outbound flight",
            "flight_start_frame": outbound_start.frame,
            "flight_start_x": outbound_start.x,
            "flight_start_y": outbound_start.y,
            "contact_to_flight_gap_frames": outbound_start.frame - contact.frame,
            "contact_launch_speed_px_per_frame": transition["launch_speed"],
            "toss": toss,
        }
    )
    return serve


def classify_serve_type(
    toss: dict[str, Any],
    flight_stats: dict[str, float],
    args: argparse.Namespace,
) -> tuple[str, str]:
    partial_toss = not bool(toss["complete_arc"])
    if toss["vertical_range_px"] >= float(args.serve_power_toss_min_height):
        return "power", "high compact vertical toss"
    if (
        partial_toss
        and toss["vertical_range_px"] >= float(args.serve_power_toss_min_height) * 0.5
        and flight_stats["avg_speed"] >= float(args.serve_power_min_speed)
    ):
        return "power", "partial compact toss plus fast outbound flight"
    return "floater", "compact low toss without power-serve height/speed evidence"


def compact_toss_suffix(
    points: list[BallPoint],
    end_frame: int,
    max_frames: int,
    max_x_drift: float,
) -> list[BallPoint]:
    """Return the longest recent suffix that remains in a narrow x column."""

    eligible = [
        point
        for point in points
        if end_frame - max_frames <= point.frame <= end_frame
    ]
    if not eligible:
        return []
    best = [eligible[-1]]
    for start in range(len(eligible) - 1, -1, -1):
        suffix = eligible[start:]
        xs = [point.x for point in suffix]
        if max(xs) - min(xs) <= max_x_drift:
            best = suffix
        else:
            break
    return best


def serve_toss_stats(points: list[BallPoint]) -> dict[str, Any]:
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    apex_index = min(range(len(points)), key=lambda index: points[index].y)
    apex = points[apex_index]
    rise = max(0.0, points[0].y - apex.y)
    fall = max(0.0, points[-1].y - apex.y)
    vertical_motion = sum(abs(cur.y - prev.y) for prev, cur in zip(points, points[1:]))
    horizontal_motion = sum(abs(cur.x - prev.x) for prev, cur in zip(points, points[1:]))
    arc_threshold = max(3.0, (max(ys) - min(ys)) * 0.15)
    return {
        "start_frame": points[0].frame,
        "end_frame": points[-1].frame,
        "apex_frame": apex.frame,
        "point_count": len(points),
        "x_span_px": max(xs) - min(xs),
        "vertical_range_px": max(ys) - min(ys),
        "vertical_motion_px": vertical_motion,
        "horizontal_motion_px": horizontal_motion,
        "vertical_to_horizontal_motion_ratio": vertical_motion / max(1.0, horizontal_motion),
        "rise_px": rise,
        "fall_px": fall,
        "complete_arc": rise >= arc_threshold and fall >= arc_threshold,
        "phase": "up_and_down" if rise >= arc_threshold and fall >= arc_threshold else "partial",
    }


def detect_reception_candidates(points: list[BallPoint], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_frame = {point.frame: point for point in points}
    candidates = []
    for point in points:
        before = nearest_point(by_frame, point.frame - args.reception_window, -1, args.reception_window * 2)
        after = nearest_point(by_frame, point.frame + args.reception_window, 1, args.reception_window * 2)
        if before is None or after is None or before.frame >= point.frame or after.frame <= point.frame:
            continue
        if point.tracklet_id is not None and (
            before.tracklet_id != point.tracklet_id or after.tracklet_id != point.tracklet_id
        ):
            continue

        vx_before = (point.x - before.x) / max(1, point.frame - before.frame)
        vy_before = (point.y - before.y) / max(1, point.frame - before.frame)
        vx_after = (after.x - point.x) / max(1, after.frame - point.frame)
        vy_after = (after.y - point.y) / max(1, after.frame - point.frame)
        speed_before = math.hypot(vx_before, vy_before)
        speed_after = math.hypot(vx_after, vy_after)
        if speed_before < 2.0 or speed_after < 2.0:
            continue

        angle = vector_angle_change((vx_before, vy_before), (vx_after, vy_after))
        if angle < args.reception_min_angle_change:
            continue

        speed_change = abs(speed_after - speed_before) / max(speed_before, speed_after)
        vertex = estimate_direction_change_moment(by_frame, point.frame, args.reception_window)
        candidates.append(
            {
                "frame": vertex.point.frame,
                "receiver_frame": round_frame(vertex.frame_float),
                "receiver_frame_float": vertex.frame_float,
                "direction_change_visible_before_frame": vertex.visible_before_frame,
                "direction_change_visible_after_frame": vertex.visible_after_frame,
                "direction_change_used_gap_midpoint": vertex.used_gap_midpoint,
                "candidate_frame": point.frame,
                "time_sec": None,
                "x": vertex.point.x,
                "y": vertex.point.y,
                "ball_sample_x": point.x,
                "ball_sample_y": point.y,
                "angle_change_deg": angle,
                "speed_change_ratio": speed_change,
                "speed_before_px_per_frame": speed_before,
                "speed_after_px_per_frame": speed_after,
                "vy_before_px_per_frame": vy_before,
                "vy_after_px_per_frame": vy_after,
                "score": angle + speed_change * 30,
            }
        )
    candidates.sort(key=lambda item: item["frame"])
    return group_reception_candidates(candidates, min_gap_frames=args.action_min_frame_gap)


def first_reception_after(points: list[BallPoint], serve_end_frame: int, args: argparse.Namespace) -> dict[str, Any] | None:
    for candidate in detect_reception_candidates(points, args):
        if candidate["frame"] > serve_end_frame + args.reception_min_frame_gap:
            return candidate
    return None


def estimate_direction_change_point(
    by_frame: dict[int, BallPoint],
    frame: int,
    window: int,
) -> tuple[float, float]:
    point = by_frame[frame]
    before_segment = collect_segment_around_frame(by_frame, frame, -1, window)
    after_segment = collect_segment_around_frame(by_frame, frame, 1, window)
    before_line = line_from_segment(before_segment)
    after_line = line_from_segment(after_segment)
    if before_line is None or after_line is None:
        return (point.x, point.y)

    intersection = line_intersection(before_line, after_line)
    if intersection is None:
        return (point.x, point.y)

    x, y = intersection
    # Guard against unstable intersections from nearly parallel or noisy lines.
    if math.hypot(x - point.x, y - point.y) > max(160.0, window * 45.0):
        return (point.x, point.y)
    return (float(x), float(y))


def estimate_direction_change_vertex(
    by_frame: dict[int, BallPoint],
    frame: int,
    window: int,
) -> BallPoint:
    return estimate_direction_change_moment(by_frame, frame, window).point


def estimate_direction_change_moment(
    by_frame: dict[int, BallPoint],
    frame: int,
    window: int,
) -> DirectionChangeMoment:
    """Return the visible trajectory vertex, not a fitted future/past sample.

    The annotated trail is drawn as straight segments between frame samples. The
    action marker should therefore sit on the polyline bend. If the bend happens
    across a visibility gap, receiver assignment uses the midpoint frame between
    the visible-before and visible-after samples.
    """
    center = by_frame[frame]
    local_points = [
        by_frame[item]
        for item in range(frame - window * 2, frame + window * 2 + 1)
        if item in by_frame
        and (center.tracklet_id is None or by_frame[item].tracklet_id == center.tracklet_id)
    ]
    if len(local_points) < 3:
        return DirectionChangeMoment(center, float(center.frame), center.frame, center.frame, False)

    best_point = center
    best_prev = center
    best_next = center
    best_score = -1.0
    for index, cur in enumerate(local_points):
        if abs(cur.frame - frame) > window:
            continue

        prev = nearest_moving_neighbor(local_points, index, -1, min_distance=2.0)
        nxt = nearest_moving_neighbor(local_points, index, 1, min_distance=2.0)
        if prev is None or nxt is None:
            continue

        v_before = (
            (cur.x - prev.x) / max(1, cur.frame - prev.frame),
            (cur.y - prev.y) / max(1, cur.frame - prev.frame),
        )
        v_after = (
            (nxt.x - cur.x) / max(1, nxt.frame - cur.frame),
            (nxt.y - cur.y) / max(1, nxt.frame - cur.frame),
        )
        angle = vector_angle_change(v_before, v_after)
        if angle > best_score or (math.isclose(angle, best_score) and cur.frame > best_point.frame):
            best_score = angle
            best_point = cur
            best_prev = prev
            best_next = nxt

    frame_float = float(best_point.frame)
    used_gap_midpoint = False
    if best_next.frame - best_point.frame > 1:
        frame_float = (best_point.frame + best_next.frame) / 2.0
        used_gap_midpoint = True
    elif best_point.frame - best_prev.frame > 1:
        frame_float = (best_prev.frame + best_point.frame) / 2.0
        used_gap_midpoint = True

    return DirectionChangeMoment(
        point=best_point,
        frame_float=frame_float,
        visible_before_frame=best_point.frame if best_next.frame - best_point.frame > 1 else best_prev.frame,
        visible_after_frame=best_next.frame if best_next.frame - best_point.frame > 1 else best_point.frame,
        used_gap_midpoint=used_gap_midpoint,
    )


def nearest_moving_neighbor(
    points: list[BallPoint],
    index: int,
    direction: int,
    min_distance: float,
) -> BallPoint | None:
    cur = points[index]
    offset = index + direction
    while 0 <= offset < len(points):
        candidate = points[offset]
        if math.hypot(cur.x - candidate.x, cur.y - candidate.y) >= min_distance:
            return candidate
        offset += direction
    return None


def collect_segment_around_frame(
    by_frame: dict[int, BallPoint],
    frame: int,
    direction: int,
    window: int,
) -> list[BallPoint]:
    points = [by_frame[frame]]
    for offset in range(1, window * 3 + 1):
        candidate = by_frame.get(frame + offset * direction)
        if candidate is not None:
            points.append(candidate)
        if len(points) >= max(3, window):
            break
    return sorted(points, key=lambda item: item.frame)


def line_from_segment(points: list[BallPoint]) -> tuple[float, float, float, float] | None:
    if len(points) < 2:
        return None
    first = points[0]
    last = points[-1]
    dx = last.x - first.x
    dy = last.y - first.y
    if math.hypot(dx, dy) < 1e-6:
        return None
    return (first.x, first.y, dx, dy)


def line_intersection(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    ax, ay, avx, avy = a
    bx, by, bvx, bvy = b
    denom = avx * bvy - avy * bvx
    if abs(denom) < 1e-6:
        return None
    t = ((bx - ax) * bvy - (by - ay) * bvx) / denom
    return (ax + t * avx, ay + t * avy)


def find_receiver(
    player_boxes_by_frame: dict[int, list[PlayerBox]],
    ball_x: float,
    ball_y: float,
    reception_frame: float,
    frame_window: int,
    max_distance: float,
    dispute_margin: float,
    known_player_bonus: float = 110.0,
    prediction_penalty: float = 35.0,
) -> dict[str, Any] | None:
    center_frame = round_frame(reception_frame)
    best_by_track: dict[str, tuple[float, float, int, PlayerBox]] = {}
    for offset in range(-frame_window, frame_window + 1):
        frame = center_frame + offset
        for box in player_boxes_by_frame.get(frame, []):
            ax, ay = box.upper_anchor
            distance = math.hypot(ball_x - ax, ball_y - ay)
            if distance <= max_distance:
                adjusted_distance = distance
                if box.player_id:
                    adjusted_distance -= max(0.0, known_player_bonus)
                if not box.observed:
                    adjusted_distance += max(0.0, prediction_penalty)
                candidate = (adjusted_distance, distance, abs(offset), box)
                current = best_by_track.get(box.track_id)
                if current is None or candidate[:3] < current[:3]:
                    best_by_track[box.track_id] = candidate
    candidates = list(best_by_track.values())
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[:3])
    best_adjusted, best_distance, best_offset, best_box = candidates[0]
    disputed = False
    alternatives = []
    for adjusted, distance, offset, box in candidates[1:4]:
        if adjusted - best_adjusted <= dispute_margin:
            disputed = True
            alternatives.append(receiver_candidate_to_dict(box, distance, offset))
    result = receiver_candidate_to_dict(best_box, best_distance, best_offset)
    result.update(
        {
            "frame": best_box.frame,
            "action_frame_float": reception_frame,
            "action_frame_rounded": center_frame,
            "ball_x": ball_x,
            "ball_y": ball_y,
            "disputed": disputed,
            "alternatives": alternatives,
        }
    )
    return result


def find_serve_contact_player(
    cv2: Any,
    video_path: Path,
    player_boxes_by_frame: dict[int, list[PlayerBox]],
    ball_x: float,
    ball_y: float,
    contact_frame: int,
    serving_team: str,
    frame_window: int,
    max_distance: float,
    pose_classifier: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Associate a serve contact against all teams without identity bias.

    Reception scoring may intentionally contain only near-team boxes.  Serve
    contact association must not use that filtered view, and an unknown
    opponent must not lose merely because a nearby receiver has a known jersey.
    When MediaPipe is already enabled for action classification, visible wrist
    landmarks are used instead of the coarse upper-box anchor.
    """

    candidates: list[tuple[float, int, PlayerBox, dict[str, Any]]] = []
    frame_cache: dict[int, Any] = {}
    for offset in range(-max(0, frame_window), max(0, frame_window) + 1):
        frame_idx = contact_frame + offset
        for box in player_boxes_by_frame.get(frame_idx, []):
            ax, ay = box.upper_anchor
            upper_distance = math.hypot(ball_x - ax, ball_y - ay)
            anchor = {"x": ax, "y": ay, "method": "upper_box_anchor", "visibility": None}
            distance = upper_distance

            if pose_classifier is not None and upper_distance <= max_distance * 1.5:
                frame = frame_cache.get(frame_idx)
                if frame is None:
                    frame = read_video_frame(cv2, video_path, frame_idx)
                    frame_cache[frame_idx] = frame
                wrist = closest_visible_wrist_to_ball(
                    cv2=cv2,
                    frame=frame,
                    box=box,
                    ball_x=ball_x,
                    ball_y=ball_y,
                    pose_classifier=pose_classifier,
                )
                if wrist is not None:
                    distance = float(wrist["distance_px"])
                    anchor = {
                        "x": wrist["x"],
                        "y": wrist["y"],
                        "method": wrist["name"],
                        "visibility": wrist["visibility"],
                    }

            if distance <= max_distance:
                candidates.append((distance, abs(offset), box, anchor))

    if not candidates:
        return None

    if serving_team == "far":
        preferred = [item for item in candidates if item[2].team in {"opponent", "far"}]
    elif serving_team == "near":
        preferred = [item for item in candidates if item[2].team == "near"]
    else:
        preferred = []
    if preferred:
        candidates = preferred

    distance, offset, box, anchor = min(
        candidates,
        key=lambda item: (item[0], item[1], 0 if item[2].observed else 1),
    )
    result = receiver_candidate_to_dict(box, distance, offset)
    result.update(
        {
            "frame": box.frame,
            "team": box.team,
            "ball_x": ball_x,
            "ball_y": ball_y,
            "contact_anchor": anchor,
            "association_method": "all_team_pose_wrist" if anchor["method"].endswith("wrist") else "all_team_upper_box",
        }
    )
    return result


def closest_visible_wrist_to_ball(
    cv2: Any,
    frame: Any,
    box: PlayerBox,
    ball_x: float,
    ball_y: float,
    pose_classifier: dict[str, Any],
) -> dict[str, Any] | None:
    if frame is None:
        return None
    pose_result = detect_pose_for_bbox(cv2, frame, box.bbox, pose_classifier["pose_estimator"])
    if not pose_result.get("ok"):
        return None
    landmarks = pose_classifier["pose_dataset"].convert_landmarks(
        pose_result["pose_landmarks"],
        crop_origin=pose_result["crop_origin"],
        crop_shape=pose_result["crop_shape"],
        frame_shape=frame.shape,
        bbox=box.bbox,
    )
    height, width = frame.shape[:2]
    wrists = []
    for point in landmarks:
        name = str(point.get("name") or "")
        visibility = float(point.get("visibility", 0.0) or 0.0)
        if name not in {"left_wrist", "right_wrist"} or visibility < 0.15:
            continue
        x = float(point["x"]) * width
        y = float(point["y"]) * height
        wrists.append(
            {
                "name": name,
                "x": x,
                "y": y,
                "visibility": visibility,
                "distance_px": math.hypot(ball_x - x, ball_y - y),
            }
        )
    return min(wrists, key=lambda item: item["distance_px"]) if wrists else None


def receiver_candidate_to_dict(box: PlayerBox, distance: float, frame_offset: int) -> dict[str, Any]:
    x1, y1, x2, y2 = box.bbox
    ax, ay = box.upper_anchor
    return {
        "track_id": box.track_id,
        "player_id": box.player_id,
        "distance_px": distance,
        "frame_offset_abs": frame_offset,
        "upper_anchor": {"x": ax, "y": ay},
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


def round_frame(frame: float) -> int:
    return int(math.floor(frame + 0.5))


def segment_points(
    by_frame: dict[int, BallPoint],
    start_frame: int,
    window: int,
    max_gap: int,
) -> list[BallPoint]:
    segment = []
    tracklet_id = None
    frame = start_frame
    target_end = start_frame + window
    while frame <= target_end:
        point = nearest_point(by_frame, frame, 1, max_gap)
        if point is None or point.frame > target_end:
            frame += 1
            continue
        if segment and tracklet_id is not None and point.tracklet_id != tracklet_id:
            break
        if not segment or point.frame > segment[-1].frame:
            segment.append(point)
            if len(segment) == 1:
                tracklet_id = point.tracklet_id
        frame = point.frame + 1
    return segment


def trajectory_stats(points: list[BallPoint]) -> dict[str, float]:
    if len(points) < 2:
        return {"distance": 0.0, "avg_speed": 0.0, "y_speed": 0.0, "mean_angle_change": 180.0}

    distance = 0.0
    vectors = []
    for prev, cur in zip(points, points[1:]):
        dt = max(1, cur.frame - prev.frame)
        vx = (cur.x - prev.x) / dt
        vy = (cur.y - prev.y) / dt
        vectors.append((vx, vy))
        distance += math.hypot(cur.x - prev.x, cur.y - prev.y)

    angles = [
        vector_angle_change(prev, cur)
        for prev, cur in zip(vectors, vectors[1:])
        if math.hypot(*prev) > 1e-6 and math.hypot(*cur) > 1e-6
    ]
    dt_total = max(1, points[-1].frame - points[0].frame)
    return {
        "distance": distance,
        "avg_speed": distance / dt_total,
        "y_speed": (points[-1].y - points[0].y) / dt_total,
        "mean_angle_change": sum(angles) / len(angles) if angles else 0.0,
    }


def group_reception_candidates(candidates: list[dict[str, Any]], min_gap_frames: int) -> list[dict[str, Any]]:
    if not candidates:
        return []
    groups = [[candidates[0]]]
    for candidate in candidates[1:]:
        if candidate["frame"] - groups[-1][-1]["frame"] <= min_gap_frames:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    grouped = [max(group, key=lambda item: item["frame"]) for group in groups]
    return sorted(grouped, key=lambda item: item["frame"])


def smooth_points(points: list[BallPoint]) -> list[BallPoint]:
    by_frame = {point.frame: point for point in points}
    smoothed = []
    for point in points:
        neighbors = [
            by_frame[frame]
            for frame in range(point.frame - 2, point.frame + 3)
            if frame in by_frame
            and (point.tracklet_id is None or by_frame[frame].tracklet_id == point.tracklet_id)
        ]
        if len(neighbors) < 3:
            smoothed.append(point)
            continue
        xs = sorted(item.x for item in neighbors)
        ys = sorted(item.y for item in neighbors)
        smoothed.append(
            BallPoint(
                point.frame,
                xs[len(xs) // 2],
                ys[len(ys) // 2],
                point.radius,
                point.confidence,
                point.tracklet_id,
            )
        )
    return smoothed


def nearest_point(by_frame: dict[int, BallPoint], target_frame: int, direction: int, max_gap: int) -> BallPoint | None:
    for offset in range(max_gap + 1):
        point = by_frame.get(target_frame + offset * direction)
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


def box_foot(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, _y1, x2, y2 = box
    return ((x1 + x2) / 2, y2)


def box_intersects_polygon(
    box: tuple[float, float, float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    if not polygon:
        return True
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if any(point_in_polygon(point, polygon) for point in corners):
        return True
    if any(point_inside_box(point, box) for point in polygon):
        return True
    box_edges = list(zip(corners, [*corners[1:], corners[0]]))
    polygon_edges = list(zip(polygon, [*polygon[1:], polygon[0]]))
    return any(
        segments_intersect(box_a, box_b, poly_a, poly_b)
        for box_a, box_b in box_edges
        for poly_a, poly_b in polygon_edges
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


def point_inside_box(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


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
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    eps = 1e-9
    return (
        abs(o1) < eps and on_segment(a, c, b)
        or abs(o2) < eps and on_segment(a, d, b)
        or abs(o3) < eps and on_segment(c, a, d)
        or abs(o4) < eps and on_segment(c, b, d)
    )


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
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def write_annotated_video(
    cv2: Any,
    video_path: Path,
    output_path: Path,
    fps: float,
    size: tuple[int, int],
    max_frames: int,
    ball_points: list[BallPoint],
    player_boxes_by_frame: dict[int, list[PlayerBox]],
    player_tracks_by_id: dict[str, list[PlayerBox]],
    serve: dict[str, Any],
    actions: list[dict[str, Any]],
    reception_evaluation: dict[str, Any],
    reception_zones: list[dict[str, Any]],
    label_hold_frames: int,
    player_draw_max_gap: int,
    trail_length: int,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    ball_by_frame = {point.frame: point for point in ball_points}
    history: list[BallPoint] = []
    frame_idx = -1
    while frame_idx + 1 < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        ball = ball_by_frame.get(frame_idx)
        if ball is not None:
            history.append(ball)
        out = frame.copy()
        active_action = active_action_for_frame(actions, frame_idx, label_hold_frames)
        draw_reception_zones(cv2, out, reception_zones)
        draw_ball(cv2, out, frame_idx, ball, history, trail_length)
        boxes = latest_player_boxes_near_frame(player_tracks_by_id, frame_idx, player_draw_max_gap)
        if not boxes:
            boxes = player_boxes_by_frame.get(frame_idx, [])
        serve_age = frame_idx - int(serve["frame"])
        active_serve = serve if 0 <= serve_age <= label_hold_frames else None
        draw_player_boxes(cv2, out, boxes, active_action, active_serve)
        draw_events(cv2, out, frame_idx, fps, serve, actions, active_action, label_hold_frames, reception_evaluation)
        writer.write(out)
    cap.release()
    writer.release()


def draw_reception_zones(cv2: Any, frame: Any, zones: list[dict[str, Any]]) -> None:
    if not zones:
        return
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError:
        return

    overlay = frame.copy()
    for zone in zones:
        polygon = zone.get("polygon") or []
        if len(polygon) < 3:
            continue
        score = float(zone.get("score", 0.0))
        color = (0, 150, 0) if score >= 1.0 else (0, 165, 255)
        points = np.array([[int(x), int(y)] for x, y in polygon], dtype=np.int32)
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=3)
        label_x, label_y = points[0]
        draw_label(cv2, frame, f"zone {score:g}", int(label_x) + 8, int(label_y) - 8, color, 0.65)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, dst=frame)


def draw_ball(cv2: Any, frame: Any, frame_idx: int, ball: BallPoint | None, history: list[BallPoint], trail_length: int) -> None:
    recent = [
        point
        for point in history
        if 0 <= frame_idx - point.frame <= trail_length
        and (ball is None or ball.tracklet_id is None or point.tracklet_id == ball.tracklet_id)
    ][-trail_length:]
    for idx in range(1, len(recent)):
        prev = recent[idx - 1]
        cur = recent[idx]
        cv2.line(frame, (int(prev.x), int(prev.y)), (int(cur.x), int(cur.y)), (255, 0, 255), 3)
    if ball is not None:
        cv2.circle(frame, (int(ball.x), int(ball.y)), int(max(9, ball.radius + 4)), (0, 255, 255), 3)


def active_action_for_frame(actions: list[dict[str, Any]], frame_idx: int, hold: int) -> dict[str, Any] | None:
    active = [action for action in actions if 0 <= frame_idx - int(action["frame"]) <= hold]
    if not active:
        return None
    return min(active, key=lambda action: abs(frame_idx - int(action["frame"])))


def draw_player_boxes(
    cv2: Any,
    frame: Any,
    boxes: list[PlayerBox],
    active_action: dict[str, Any] | None,
    active_serve: dict[str, Any] | None = None,
) -> None:
    receiver = active_action.get("receiver") if active_action else None
    receiver_track = str(receiver.get("track_id")) if receiver else None
    receiver_pose_action = receiver.get("pose_action") if receiver else None
    contact_player = active_serve.get("contact_player") if active_serve else None
    server_track = str(contact_player.get("track_id")) if contact_player else None
    for box in boxes:
        x1, y1, x2, y2 = [int(value) for value in box.bbox]
        if receiver_track and box.track_id == receiver_track:
            color = (0, 255, 0)
        elif box.team == "opponent":
            color = (0, 0, 255)
        elif box.team == "near":
            color = (255, 160, 0)
        else:
            color = (0, 215, 255)
        thickness = 4 if server_track and box.track_id == server_track else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        ax, ay = box.upper_anchor
        cv2.circle(frame, (int(ax), int(ay)), 5, color, -1)
        if box.team == "opponent":
            label = f"opponent track_{box.track_id}"
        elif box.player_id:
            label = box.player_id
        else:
            label = f"unknown track_{box.track_id}"
        if server_track and box.track_id == server_track:
            label = f"SERVER {label}"
        if receiver_track and box.track_id == receiver_track and receiver_pose_action and receiver_pose_action.get("ok"):
            label = f"{label} {format_probability_vector(receiver_pose_action['probabilities'])}"
        draw_label(cv2, frame, label, x1, max(20, y1 - 8), color, 0.55)


def draw_events(
    cv2: Any,
    frame: Any,
    frame_idx: int,
    fps: float,
    serve: dict[str, Any],
    actions: list[dict[str, Any]],
    active_action: dict[str, Any] | None,
    hold: int,
    reception_evaluation: dict[str, Any],
) -> None:
    lines = [f"serve: {serve['serving_team']} team"]
    if reception_evaluation.get("ok"):
        lines.append(
            f"reception score: {reception_evaluation.get('score')} "
            f"receiver: {reception_evaluation.get('receiver') or 'none'}"
        )
    elif reception_evaluation:
        lines.append("reception score: unavailable")
    if active_action is not None:
        action_line = f"{active_action.get('action_type', 'action')}: {active_action['frame'] / fps:.2f}s"
        if active_action.get("role") == "reception" and active_action.get("reception_score") is not None:
            action_line += f" score={active_action['reception_score']}"
        lines.append(action_line)
    receiver = active_action.get("receiver") if active_action else None
    if receiver is not None:
        prefix = "DISPUTED " if receiver["disputed"] else ""
        lines.append(f"{prefix}receiver: {receiver.get('player_id') or receiver.get('track_id')}")
        pose_action = receiver.get("pose_action")
        if pose_action and pose_action.get("ok"):
            lines.append(f"action p: {format_probability_vector(pose_action['probabilities'])}")
    contact_player = serve.get("contact_player") or {}
    if contact_player:
        server_label = contact_player.get("player_id") or f"track_{contact_player.get('track_id')}"
        lines.append(f"server: {server_label} ({contact_player.get('team', 'unknown')})")
    for idx, line in enumerate(lines):
        draw_label(cv2, frame, line, 20, 35 + idx * 28, (0, 0, 255), 0.8)

    if 0 <= frame_idx - int(serve["frame"]) <= hold:
        cv2.drawMarker(frame, (int(serve["x"]), int(serve["y"])), (255, 120, 0), cv2.MARKER_TILTED_CROSS, 42, 4)
        serve_type = str(serve.get("serve_type") or "unknown").upper()
        label = f"{serve_type} SERVE" if serve_type != "UNKNOWN" else "SERVE FLIGHT START"
        if not serve.get("contact_captured", False):
            label += " (CONTACT NOT CAPTURED)"
        draw_label(cv2, frame, label, int(serve["x"]) + 12, int(serve["y"]), (255, 120, 0), 0.75)
    for action in actions:
        if 0 <= frame_idx - int(action["frame"]) <= hold:
            if action.get("role") == "reception":
                color = (0, 0, 255)
                label = f"RECEPTION {action.get('reception_score', '')}".strip()
            elif action.get("role") == "pass":
                color = (255, 80, 0)
                label = "PASS"
            else:
                color = (0, 180, 255) if action.get("action_type") == "wait" else (0, 0, 255)
                label = str(action.get("action_type") or "ACTION").upper()
            cv2.drawMarker(frame, (int(action["x"]), int(action["y"])), color, cv2.MARKER_CROSS, 44, 4)
            draw_label(cv2, frame, label, int(action["x"]) + 12, int(action["y"]), color, 0.75)

    pass_info = reception_evaluation.get("pass") or {}
    origin = pass_info.get("origin_court_anchor") or {}
    if origin:
        origin_frame = int(origin.get("frame", pass_info.get("frame", 0)))
        pass_frame = int(pass_info.get("frame", origin_frame))
        if origin_frame <= frame_idx <= pass_frame + hold:
            x, y = int(round(float(origin["x"]))), int(round(float(origin["y"])))
            color = (255, 0, 255)
            cv2.circle(frame, (x, y), 12, color, 3, cv2.LINE_AA)
            cv2.drawMarker(frame, (x, y), color, cv2.MARKER_TILTED_CROSS, 26, 3)
            draw_label(cv2, frame, "SETTER PASS/TAKEOFF POINT", x + 16, y - 12, color, 0.65)


def draw_label(cv2: Any, frame: Any, text: str, x: int, y: int, color: tuple[int, int, int], scale: float) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 4)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def format_probability_vector(probabilities: dict[str, float]) -> str:
    return " ".join(f"{label}={probability:.2f}" for label, probability in probabilities.items())


if __name__ == "__main__":
    raise SystemExit(main())
