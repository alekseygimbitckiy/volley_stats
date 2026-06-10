#!/usr/bin/env python3
"""Batch-build a player-pose action dataset from all manually segmented folders."""

from __future__ import annotations

import argparse
import json
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import build_action_pose_dataset as ui


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = ROOT / "data" / "raw_labeling"
DEFAULT_COURTS_DIR = ROOT / "data" / "processed" / "action_pose_dataset" / "courts"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "action_pose_dataset_batch"


class BatchContext:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = ROOT
        self.video_dir = ui.resolve_path(args.video_dir)
        self.courts_dir = ui.resolve_path(args.courts_dir)
        self.output_dir = ui.resolve_path(args.output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.csv_path = self.output_dir / "samples.csv"
        self.jsonl_path = self.output_dir / "samples.jsonl"
        self.lock = threading.Lock()
        self._deps: dict[str, Any] | None = None
        self._yolo_model: Any | None = None

    def deps(self) -> dict[str, Any]:
        if self._deps is None:
            self._deps = ui.import_dependencies()
        return self._deps

    def yolo_model(self) -> Any:
        if self._yolo_model is None:
            model_path = str(ui.resolve_path(self.args.yolo_model))
            print(f"Loading YOLO model: {model_path}")
            self._yolo_model = ui.import_yolo()(model_path)
        return self._yolo_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process all videos with saved folder-level court polygons into a pose classification dataset."
    )
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR.relative_to(ROOT)))
    parser.add_argument("--courts-dir", default=str(DEFAULT_COURTS_DIR.relative_to(ROOT)))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT)))
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--person-conf", type=float, default=0.35)
    parser.add_argument("--person-iou", type=float, default=0.55)
    parser.add_argument("--max-player-bbox-iou", type=float, default=0.75)
    parser.add_argument("--frame-stride", type=int, default=1, help="1 means process every frame.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process each whole video.")
    parser.add_argument("--flush-every", type=int, default=250, help="Frames between dataset writes.")
    parser.add_argument("--label-source", choices=["stem", "name", "parent"], default="stem")
    parser.add_argument("--lower-band-ratio", type=float, default=0.35)
    parser.add_argument("--min-lower-court-intersection", type=float, default=0.08)
    parser.add_argument("--pose-min-detection-confidence", type=float, default=0.35)
    parser.add_argument("--pose-min-visible-landmarks", type=int, default=10)
    parser.add_argument("--court-foot-landmarks", default="left_foot_index,right_foot_index")
    parser.add_argument("--min-court-foot-visibility", type=float, default=0.25)
    parser.add_argument("--min-court-feet-inside", type=int, default=1)
    parser.add_argument("--pose-model", default="external/mediapipe/pose_landmarker_lite.task")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing samples.csv and samples.jsonl in the output directory before processing.",
    )
    args = parser.parse_args()

    args.frame_stride = max(1, args.frame_stride)
    args.flush_every = max(1, args.flush_every)

    context = BatchContext(args)
    if not context.video_dir.exists():
        raise SystemExit(f"Video directory not found: {context.video_dir}")
    if not context.courts_dir.exists():
        raise SystemExit(f"Courts directory not found: {context.courts_dir}")

    context.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in (context.csv_path, context.jsonl_path):
            if path.exists():
                path.unlink()

    videos = ui.list_videos(context.video_dir)
    if not videos:
        raise SystemExit(f"No videos found in {context.video_dir}")

    deps = context.deps()
    mp = ui.import_mediapipe()
    pose = ui.create_pose_estimator(mp, deps["np"], args)
    court_landmarks = ui.parse_landmark_names(args.court_foot_landmarks)

    total_samples = 0
    total_frames = 0
    skipped_no_court = 0
    try:
        for video in videos:
            video_path = context.video_dir / video["relative_path"]
            court_path = ui.court_file(context.courts_dir, context.video_dir, video_path)
            if not court_path.exists():
                skipped_no_court += 1
                print(f"Skipping without court: {video['relative_path']} expected {court_path}")
                continue
            polygon = load_court_polygon(court_path)
            label = video_label(video_path, args.label_source)
            stats = process_video(
                context=context,
                video_path=video_path,
                polygon=polygon,
                action_label=label,
                pose=pose,
                court_landmarks=court_landmarks,
            )
            total_samples += stats["samples"]
            total_frames += stats["frames"]
    finally:
        pose.close()

    print("Done.")
    print(f"Processed frames: {total_frames}")
    print(f"Saved samples: {total_samples}")
    print(f"Videos skipped without court: {skipped_no_court}")
    print(f"CSV: {context.csv_path}")
    print(f"JSONL: {context.jsonl_path}")
    return 0


def process_video(
    context: BatchContext,
    video_path: Path,
    polygon: list[tuple[float, float]],
    action_label: str,
    pose: ui.PoseEstimator,
    court_landmarks: list[str],
) -> dict[str, int]:
    cv2 = context.deps()["cv2"]
    np = context.deps()["np"]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: could not open {video_path}")
        return {"frames": 0, "samples": 0}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if context.args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, context.args.start_frame)

    pending: list[ui.PlayerSample] = []
    frames_processed = 0
    samples_saved = 0
    frame_idx = max(0, context.args.start_frame) - 1
    max_frames = context.args.max_frames if context.args.max_frames > 0 else None

    print(f"Processing {video_path.relative_to(context.video_dir)} label={action_label} frames={frame_count}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if max_frames is not None and frames_processed >= max_frames:
            break
        if frame_idx % context.args.frame_stride != 0:
            continue

        samples = process_frame_image(
            context=context,
            frame=frame,
            video_path=video_path,
            frame_index=frame_idx,
            polygon=polygon,
            action_label=action_label,
            pose=pose,
            court_landmarks=court_landmarks,
        )
        pending.extend(samples)
        frames_processed += 1
        samples_saved += len(samples)

        if pending and frames_processed % context.args.flush_every == 0:
            ui.append_samples(context, pending)
            pending.clear()

        if frames_processed % 100 == 0:
            print(f"  frame {frame_idx}: processed={frames_processed} samples={samples_saved}")

    cap.release()
    if pending:
        ui.append_samples(context, pending)
    print(f"  saved {samples_saved} samples from {frames_processed} processed frames")
    return {"frames": frames_processed, "samples": samples_saved}


def process_frame_image(
    context: BatchContext,
    frame: Any,
    video_path: Path,
    frame_index: int,
    polygon: list[tuple[float, float]],
    action_label: str,
    pose: ui.PoseEstimator,
    court_landmarks: list[str],
) -> list[ui.PlayerSample]:
    cv2 = context.deps()["cv2"]
    np = context.deps()["np"]
    height, width = frame.shape[:2]
    result = context.yolo_model().predict(
        frame,
        classes=[0],
        conf=context.args.person_conf,
        iou=context.args.person_iou,
        device=context.args.device,
        verbose=False,
    )[0]

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    court_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(court_mask, np.array([polygon], dtype=np.int32), 255)
    person_boxes = [
        tuple(float(value) for value in boxes.xyxy[idx].cpu().tolist())
        for idx in range(len(boxes))
    ]

    samples: list[ui.PlayerSample] = []
    for player_index, bbox in enumerate(person_boxes):
        x1, y1, x2, y2 = bbox
        if ui.max_bbox_iou(bbox, person_boxes, player_index) > context.args.max_player_bbox_iou:
            continue

        crop, crop_origin, _expanded_bbox = ui.crop_player(frame, bbox)
        if crop.size == 0:
            continue
        pose_landmarks = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not pose_landmarks:
            continue

        landmarks = ui.convert_landmarks(
            pose_landmarks,
            crop_origin=crop_origin,
            crop_shape=crop.shape,
            frame_shape=frame.shape,
            bbox=bbox,
        )
        visible = sum(1 for point in landmarks if float(point["visibility"]) >= 0.35)
        if visible < context.args.pose_min_visible_landmarks:
            continue

        foot_points_on_court, checked_foot_points = ui.count_landmarks_on_court(
            court_mask=court_mask,
            frame_shape=frame.shape,
            landmarks=landmarks,
            landmark_names=court_landmarks,
            min_visibility=context.args.min_court_foot_visibility,
        )
        if foot_points_on_court < context.args.min_court_feet_inside:
            continue

        conf = float(boxes.conf[player_index].cpu().item())
        lower_intersection = ui.lower_bbox_court_intersection(
            np=np,
            court_mask=court_mask,
            bbox=bbox,
            lower_band_ratio=context.args.lower_band_ratio,
        )
        sample_number = len(samples) + 1
        samples.append(
            ui.PlayerSample(
                sample_id=f"{video_path.stem}_f{frame_index:06d}_p{sample_number:02d}_{uuid.uuid4().hex[:8]}",
                video_name=video_path.name,
                video_path=str(video_path.relative_to(context.root)),
                frame_index=frame_index,
                action_label=action_label,
                player_index=sample_number,
                yolo_confidence=conf,
                bbox=bbox,
                lower_court_intersection=lower_intersection,
                foot_court_score=foot_points_on_court / max(1, checked_foot_points),
                foot_points_on_court=foot_points_on_court,
                visible_landmarks=visible,
                landmarks=landmarks,
            )
        )
    return samples


def load_court_polygon(path: Path) -> list[tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ui.validate_polygon(data.get("court_polygon"))


def video_label(video_path: Path, source: str) -> str:
    if source == "name":
        return video_path.name
    if source == "parent":
        return video_path.parent.name
    return video_path.stem


if __name__ == "__main__":
    raise SystemExit(main())
