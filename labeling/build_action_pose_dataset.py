#!/usr/bin/env python3
"""Build a player-pose action classification dataset from raw volleyball videos.

Run this script, open the local web UI, choose a video, mark the court on a
random frame, then process frames. Each accepted on-court player becomes one
dataset sample with YOLO bbox metadata and MediaPipe pose landmarks.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import threading
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = ROOT / "data" / "raw_labeling"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "action_pose_dataset"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

LANDMARK_NAMES = [
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


@dataclass
class PlayerSample:
    sample_id: str
    video_name: str
    video_path: str
    frame_index: int
    action_label: str
    player_index: int
    yolo_confidence: float
    bbox: tuple[float, float, float, float]
    lower_court_intersection: float
    foot_court_score: float
    foot_points_on_court: int
    visible_landmarks: int
    landmarks: list[dict[str, float | str]]


class DatasetBuilderServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], args: argparse.Namespace) -> None:
        super().__init__(address, DatasetBuilderHandler)
        self.args = args
        self.root = ROOT
        self.video_dir = resolve_path(args.video_dir)
        self.output_dir = resolve_path(args.output_dir)
        self.courts_dir = self.output_dir / "courts"
        self.frames_dir = self.output_dir / "frames"
        self.csv_path = self.output_dir / "samples.csv"
        self.jsonl_path = self.output_dir / "samples.jsonl"
        self.lock = threading.Lock()
        self._deps: dict[str, Any] | None = None
        self._yolo_model: Any | None = None

    def deps(self) -> dict[str, Any]:
        if self._deps is None:
            self._deps = import_dependencies()
        return self._deps

    def yolo_model(self) -> Any:
        if self._yolo_model is None:
            model_path = str(resolve_path(self.args.yolo_model))
            print(f"Loading YOLO model: {model_path}")
            self._yolo_model = import_yolo()(model_path)
        return self._yolo_model


class DatasetBuilderHandler(BaseHTTPRequestHandler):
    server: DatasetBuilderServer
    server_version = "ActionPoseDatasetUI/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html()
            return
        if parsed.path == "/api/videos":
            self._send_json({"videos": list_videos(self.server.video_dir)})
            return
        if parsed.path == "/api/random-frame":
            self._handle_random_frame(parsed.query)
            return
        if parsed.path == "/api/court":
            self._handle_get_court(parsed.query)
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/process-frame":
            self._handle_process_frame()
            return
        if parsed.path == "/api/save-court":
            self._handle_save_court()
            return
        self.send_error(404, "Not found")

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _handle_random_frame(self, query: str) -> None:
        params = parse_qs(query)
        video_rel = first_param(params, "video")
        try:
            video_path = safe_video_path(self.server.video_dir, video_rel)
            frame = read_random_frame(self.server.deps()["cv2"], self.server.video_dir, video_path)
        except Exception as exc:  # noqa: BLE001 - surfaced to browser.
            self._send_json({"ok": False, "error": str(exc)}, status=400)
            return
        self._send_json({"ok": True, **frame})

    def _handle_get_court(self, query: str) -> None:
        params = parse_qs(query)
        video_rel = first_param(params, "video")
        try:
            video_path = safe_video_path(self.server.video_dir, video_rel)
            court_path = court_file(self.server.courts_dir, self.server.video_dir, video_path)
            if not court_path.exists():
                self._send_json({"ok": True, "court_polygon": []})
                return
            self._send_json({"ok": True, **json.loads(court_path.read_text(encoding="utf-8"))})
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _handle_save_court(self) -> None:
        try:
            payload = self._read_json()
            video_path = safe_video_path(self.server.video_dir, str(payload["video"]))
            polygon = validate_polygon(payload["court_polygon"])
            self.server.courts_dir.mkdir(parents=True, exist_ok=True)
            court_path = court_file(self.server.courts_dir, self.server.video_dir, video_path)
            court_path.write_text(
                json.dumps(
                    {
                        "court_group": court_group(self.server.video_dir, video_path),
                        "source_video_name": video_path.name,
                        "source_video_path": str(video_path.relative_to(self.server.root)),
                        "court_polygon": polygon,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, status=400)
            return
        self._send_json({"ok": True, "court_file": str(court_path)})

    def _handle_process_frame(self) -> None:
        try:
            payload = self._read_json()
            video_path = safe_video_path(self.server.video_dir, str(payload["video"]))
            frame_index = int(payload["frame_index"])
            polygon = validate_polygon(payload["court_polygon"])
            action_label = str(payload.get("action_label") or "").strip() or video_path.stem
            samples, annotated_png = process_frame(
                server=self.server,
                video_path=video_path,
                frame_index=frame_index,
                polygon=polygon,
                action_label=action_label,
            )
            if samples:
                with self.server.lock:
                    append_samples(self.server, samples)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, status=400)
            return
        self._send_json(
            {
                "ok": True,
                "saved_samples": len(samples),
                "csv": str(self.server.csv_path),
                "jsonl": str(self.server.jsonl_path),
                "annotated_png": annotated_png,
                "samples": [sample_to_response(sample) for sample in samples],
            }
        )

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_html(self) -> None:
        content = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        content = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a pose dataset for volleyball action classification.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR.relative_to(ROOT)))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT)))
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--device", default="cpu", help="Ultralytics device: cpu, 0, cuda:0, etc.")
    parser.add_argument("--person-conf", type=float, default=0.35)
    parser.add_argument("--person-iou", type=float, default=0.55)
    parser.add_argument(
        "--max-player-bbox-iou",
        type=float,
        default=0.75,
        help="Skip a detected player if their bbox IoU with any other person bbox is above this threshold.",
    )
    parser.add_argument(
        "--lower-band-ratio",
        type=float,
        default=0.35,
        help="Bottom fraction of each bbox used for court-intersection filtering.",
    )
    parser.add_argument(
        "--min-lower-court-intersection",
        type=float,
        default=0.08,
        help="Legacy metadata threshold; toe landmarks are used for court filtering.",
    )
    parser.add_argument("--pose-min-detection-confidence", type=float, default=0.35)
    parser.add_argument("--pose-min-visible-landmarks", type=int, default=10)
    parser.add_argument(
        "--court-foot-landmarks",
        default="left_foot_index,right_foot_index",
        help="Comma-separated pose landmarks used to decide whether the player stands on the court.",
    )
    parser.add_argument(
        "--min-court-foot-visibility",
        type=float,
        default=0.25,
        help="Minimum MediaPipe visibility for a toe/foot landmark to count for court filtering.",
    )
    parser.add_argument(
        "--min-court-feet-inside",
        type=int,
        default=1,
        help="Minimum number of selected toe/foot landmarks that must be inside the marked court.",
    )
    parser.add_argument(
        "--pose-model",
        default="external/mediapipe/pose_landmarker_lite.task",
        help="MediaPipe Tasks pose landmarker model. Only needed for MediaPipe versions without mp.solutions.",
    )
    args = parser.parse_args()

    if not resolve_path(args.video_dir).exists():
        raise SystemExit(f"Video directory not found: {resolve_path(args.video_dir)}")

    server = DatasetBuilderServer((args.host, args.port), args)
    print(f"Action pose dataset UI: http://{args.host}:{args.port}/")
    print(f"Videos: {server.video_dir}")
    print(f"Outputs: {server.output_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping action pose dataset UI.")
    finally:
        server.server_close()
    return 0


def import_dependencies() -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
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
    if missing:
        raise RuntimeError(
            "Missing packages: "
            + ", ".join(missing)
            + "\nInstall with: ./venv/bin/python -m pip install "
            + " ".join(missing)
        )
    return {"cv2": cv2, "np": np}


def import_yolo() -> Any:
    try:
        from ultralytics import YOLO  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing package: ultralytics\nInstall with: ./venv/bin/python -m pip install ultralytics") from exc
    return YOLO


def import_mediapipe() -> Any:
    try:
        import mediapipe as mp  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing package: mediapipe\nInstall with: ./venv/bin/python -m pip install mediapipe") from exc
    return mp


class PoseEstimator:
    def close(self) -> None:
        pass

    def process(self, rgb: Any) -> Any | None:
        raise NotImplementedError


class LegacySolutionsPoseEstimator(PoseEstimator):
    def __init__(self, mp: Any, min_detection_confidence: float) -> None:
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
        )

    def process(self, rgb: Any) -> Any | None:
        result = self.pose.process(rgb)
        return result.pose_landmarks.landmark if result.pose_landmarks else None

    def close(self) -> None:
        self.pose.close()


class TasksPoseEstimator(PoseEstimator):
    def __init__(self, mp: Any, np: Any, model_path: Path, min_detection_confidence: float) -> None:
        if not model_path.exists():
            raise RuntimeError(
                f"MediaPipe pose model not found: {model_path}\n"
                "Download it with:\n"
                "  mkdir -p external/mediapipe\n"
                "  wget -O external/mediapipe/pose_landmarker_lite.task "
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task\n"
                "or pass --pose-model /path/to/pose_landmarker.task"
            )

        from mediapipe.tasks import python  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore

        self.mp = mp
        self.np = np
        self.landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence,
                output_segmentation_masks=False,
            )
        )

    def process(self, rgb: Any) -> Any | None:
        image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=self.np.ascontiguousarray(rgb),
        )
        result = self.landmarker.detect(image)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]

    def close(self) -> None:
        self.landmarker.close()


def create_pose_estimator(mp: Any, np: Any, args: argparse.Namespace) -> PoseEstimator:
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return LegacySolutionsPoseEstimator(mp, args.pose_min_detection_confidence)
    return TasksPoseEstimator(
        mp=mp,
        np=np,
        model_path=resolve_path(args.pose_model),
        min_detection_confidence=args.pose_min_detection_confidence,
    )


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def list_videos(video_dir: Path) -> list[dict[str, Any]]:
    videos = []
    for path in sorted(video_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(
                {
                    "name": path.name,
                    "relative_path": str(path.relative_to(video_dir)),
                    "project_path": str(path.relative_to(ROOT)),
                    "size_mb": round(path.stat().st_size / 1_000_000, 2),
                }
            )
    return videos


def first_param(params: dict[str, list[str]], name: str) -> str:
    value = params.get(name, [""])[0]
    if not value:
        raise ValueError(f"Missing query parameter: {name}")
    return value


def safe_video_path(video_dir: Path, video_rel: str) -> Path:
    path = (video_dir / video_rel).resolve()
    if not path.is_file() or video_dir.resolve() not in path.parents:
        raise ValueError(f"Video is outside {video_dir}: {video_rel}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {path.name}")
    return path


def read_random_frame(cv2: Any, video_dir: Path, video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0:
        raise ValueError(f"Could not read frame count for: {video_path}")
    frame_index = random.randint(0, max(0, frame_count - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    return {
        "video": str(video_path.relative_to(video_dir)),
        "video_name": video_path.name,
        "frame_index": frame_index,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "image_png": encode_png(cv2, frame),
    }


def read_frame(cv2: Any, video_path: Path, frame_index: int) -> Any:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def validate_polygon(raw: Any) -> list[tuple[float, float]]:
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError("Mark at least 3 court points before processing.")
    polygon = []
    for point in raw:
        if isinstance(point, dict):
            polygon.append((float(point["x"]), float(point["y"])))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            polygon.append((float(point[0]), float(point[1])))
        else:
            raise ValueError(f"Invalid court point: {point!r}")
    return polygon


def process_frame(
    server: DatasetBuilderServer,
    video_path: Path,
    frame_index: int,
    polygon: list[tuple[float, float]],
    action_label: str,
) -> tuple[list[PlayerSample], str]:
    deps = server.deps()
    cv2 = deps["cv2"]
    np = deps["np"]
    mp = import_mediapipe()
    frame = read_frame(cv2, video_path, frame_index)
    height, width = frame.shape[:2]

    result = server.yolo_model().predict(
        frame,
        classes=[0],
        conf=server.args.person_conf,
        iou=server.args.person_iou,
        device=server.args.device,
        verbose=False,
    )[0]

    court_mask = np.zeros((height, width), dtype=np.uint8)
    polygon_np = np.array([polygon], dtype=np.int32)
    cv2.fillPoly(court_mask, polygon_np, 255)

    annotated = frame.copy()
    cv2.polylines(annotated, polygon_np, isClosed=True, color=(40, 180, 80), thickness=3)

    samples = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return [], encode_png(cv2, annotated)

    pose = create_pose_estimator(mp, np, server.args)
    court_landmarks = parse_landmark_names(server.args.court_foot_landmarks)
    person_boxes = [
        tuple(float(value) for value in boxes.xyxy[idx].cpu().tolist())
        for idx in range(len(boxes))
    ]
    try:
        for player_index in range(len(boxes)):
            x1, y1, x2, y2 = person_boxes[player_index]
            conf = float(boxes.conf[player_index].cpu().item())
            max_overlap = max_bbox_iou(person_boxes[player_index], person_boxes, player_index)
            if max_overlap > server.args.max_player_bbox_iou:
                draw_bbox(cv2, annotated, (x1, y1, x2, y2), (90, 90, 220), f"overlap {max_overlap:.2f}")
                continue
            intersection = lower_bbox_court_intersection(
                np=np,
                court_mask=court_mask,
                bbox=(x1, y1, x2, y2),
                lower_band_ratio=server.args.lower_band_ratio,
            )

            crop, crop_origin, expanded_bbox = crop_player(frame, (x1, y1, x2, y2))
            if crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pose_landmarks = pose.process(rgb)
            if not pose_landmarks:
                draw_bbox(cv2, annotated, (x1, y1, x2, y2), (0, 170, 255), "no pose")
                continue

            landmarks = convert_landmarks(
                pose_landmarks,
                crop_origin=crop_origin,
                crop_shape=crop.shape,
                frame_shape=frame.shape,
                bbox=(x1, y1, x2, y2),
            )
            visible = sum(1 for point in landmarks if float(point["visibility"]) >= 0.35)
            if visible < server.args.pose_min_visible_landmarks:
                draw_bbox(cv2, annotated, (x1, y1, x2, y2), (0, 170, 255), f"weak pose {visible}")
                continue
            foot_points_on_court, checked_foot_points = count_landmarks_on_court(
                court_mask=court_mask,
                frame_shape=frame.shape,
                landmarks=landmarks,
                landmark_names=court_landmarks,
                min_visibility=server.args.min_court_foot_visibility,
            )
            foot_court_score = foot_points_on_court / max(1, checked_foot_points)
            if foot_points_on_court < server.args.min_court_feet_inside:
                draw_bbox(
                    cv2,
                    annotated,
                    expanded_bbox,
                    (150, 150, 150),
                    f"off toes {foot_points_on_court}/{checked_foot_points}",
                )
                draw_pose(cv2, annotated, landmarks)
                continue

            sample_id = f"{video_path.stem}_f{frame_index:06d}_p{len(samples) + 1:02d}_{uuid.uuid4().hex[:8]}"
            samples.append(
                PlayerSample(
                    sample_id=sample_id,
                    video_name=video_path.name,
                    video_path=str(video_path.relative_to(server.root)),
                    frame_index=frame_index,
                    action_label=action_label,
                    player_index=len(samples) + 1,
                    yolo_confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    lower_court_intersection=intersection,
                    foot_court_score=foot_court_score,
                    foot_points_on_court=foot_points_on_court,
                    visible_landmarks=visible,
                    landmarks=landmarks,
                )
            )
            draw_bbox(cv2, annotated, expanded_bbox, (35, 145, 245), f"saved toes {foot_points_on_court}/{checked_foot_points}")
            draw_pose(cv2, annotated, landmarks)
    finally:
        pose.close()

    if samples:
        server.frames_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(server.frames_dir / f"{video_path.stem}_f{frame_index:06d}.jpg"), frame)
    return samples, encode_png(cv2, annotated)


def lower_bbox_court_intersection(
    np: Any,
    court_mask: Any,
    bbox: tuple[float, float, float, float],
    lower_band_ratio: float,
) -> float:
    x1, y1, x2, y2 = bbox
    h, w = court_mask.shape[:2]
    ix1 = max(0, min(w - 1, int(round(x1))))
    ix2 = max(0, min(w, int(round(x2))))
    iy2 = max(0, min(h, int(round(y2))))
    band_top = y1 + (y2 - y1) * (1.0 - lower_band_ratio)
    iy1 = max(0, min(h - 1, int(round(band_top))))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    band = court_mask[iy1:iy2, ix1:ix2]
    return float(np.count_nonzero(band)) / float(band.size or 1)


def max_bbox_iou(
    bbox: tuple[float, float, float, float],
    bboxes: list[tuple[float, float, float, float]],
    own_index: int,
) -> float:
    overlaps = [bbox_iou(bbox, other) for idx, other in enumerate(bboxes) if idx != own_index]
    return max(overlaps, default=0.0)


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
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


def parse_landmark_names(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one --court-foot-landmarks value is required.")
    unknown = [name for name in names if name not in LANDMARK_NAMES]
    if unknown:
        raise ValueError(f"Unknown court foot landmarks: {', '.join(unknown)}")
    return names


def count_landmarks_on_court(
    court_mask: Any,
    frame_shape: tuple[int, ...],
    landmarks: list[dict[str, float | str]],
    landmark_names: list[str],
    min_visibility: float,
) -> tuple[int, int]:
    h, w = frame_shape[:2]
    by_name = {str(point["name"]): point for point in landmarks}
    inside = 0
    checked = 0
    for name in landmark_names:
        point = by_name.get(name)
        if point is None or float(point["visibility"]) < min_visibility:
            continue
        checked += 1
        x = max(0, min(w - 1, int(round(float(point["x"]) * w))))
        y = max(0, min(h - 1, int(round(float(point["y"]) * h))))
        if int(court_mask[y, x]) > 0:
            inside += 1
    return inside, checked


def crop_player(frame: Any, bbox: tuple[float, float, float, float]) -> tuple[Any, tuple[int, int], tuple[float, float, float, float]]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = 0.12 * (x2 - x1)
    pad_y = 0.08 * (y2 - y1)
    ix1 = max(0, int(round(x1 - pad_x)))
    iy1 = max(0, int(round(y1 - pad_y)))
    ix2 = min(w, int(round(x2 + pad_x)))
    iy2 = min(h, int(round(y2 + pad_y)))
    return frame[iy1:iy2, ix1:ix2], (ix1, iy1), (float(ix1), float(iy1), float(ix2), float(iy2))


def convert_landmarks(
    raw_landmarks: Any,
    crop_origin: tuple[int, int],
    crop_shape: tuple[int, ...],
    frame_shape: tuple[int, ...],
    bbox: tuple[float, float, float, float],
) -> list[dict[str, float | str]]:
    crop_h, crop_w = crop_shape[:2]
    frame_h, frame_w = frame_shape[:2]
    ox, oy = crop_origin
    x1, y1, x2, y2 = bbox
    bbox_w = max(1.0, x2 - x1)
    bbox_h = max(1.0, y2 - y1)
    points = []
    for idx, landmark in enumerate(raw_landmarks):
        image_x = ox + float(landmark.x) * crop_w
        image_y = oy + float(landmark.y) * crop_h
        points.append(
            {
                "name": LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"landmark_{idx}",
                "x": round(image_x / frame_w, 8),
                "y": round(image_y / frame_h, 8),
                "z": round(float(landmark.z), 8),
                "visibility": round(float(landmark.visibility), 8),
                "bbox_x": round((image_x - x1) / bbox_w, 8),
                "bbox_y": round((image_y - y1) / bbox_h, 8),
            }
        )
    return points


def append_samples(server: DatasetBuilderServer, samples: list[PlayerSample]) -> None:
    server.output_dir.mkdir(parents=True, exist_ok=True)
    write_header = not server.csv_path.exists()
    with server.csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames())
        if write_header:
            writer.writeheader()
        for sample in samples:
            writer.writerow(sample_to_csv_row(sample))
    with server.jsonl_path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample_to_json(sample), ensure_ascii=False) + "\n")


def csv_fieldnames() -> list[str]:
    fields = [
        "sample_id",
        "video_name",
        "video_path",
        "frame_index",
        "action_label",
        "player_index",
        "yolo_confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "lower_court_intersection",
        "foot_court_score",
        "foot_points_on_court",
        "visible_landmarks",
    ]
    for name in LANDMARK_NAMES:
        fields.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility", f"{name}_bbox_x", f"{name}_bbox_y"])
    return fields


def sample_to_csv_row(sample: PlayerSample) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "video_name": sample.video_name,
        "video_path": sample.video_path,
        "frame_index": sample.frame_index,
        "action_label": sample.action_label,
        "player_index": sample.player_index,
        "yolo_confidence": round(sample.yolo_confidence, 6),
        "x1": round(sample.bbox[0], 3),
        "y1": round(sample.bbox[1], 3),
        "x2": round(sample.bbox[2], 3),
        "y2": round(sample.bbox[3], 3),
        "lower_court_intersection": round(sample.lower_court_intersection, 6),
        "foot_court_score": round(sample.foot_court_score, 6),
        "foot_points_on_court": sample.foot_points_on_court,
        "visible_landmarks": sample.visible_landmarks,
    }
    by_name = {str(point["name"]): point for point in sample.landmarks}
    for name in LANDMARK_NAMES:
        point = by_name.get(name, {})
        row[f"{name}_x"] = point.get("x", "")
        row[f"{name}_y"] = point.get("y", "")
        row[f"{name}_z"] = point.get("z", "")
        row[f"{name}_visibility"] = point.get("visibility", "")
        row[f"{name}_bbox_x"] = point.get("bbox_x", "")
        row[f"{name}_bbox_y"] = point.get("bbox_y", "")
    return row


def sample_to_json(sample: PlayerSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "video_name": sample.video_name,
        "video_path": sample.video_path,
        "frame_index": sample.frame_index,
        "action_label": sample.action_label,
        "player_index": sample.player_index,
        "yolo_confidence": sample.yolo_confidence,
        "bbox": {"x1": sample.bbox[0], "y1": sample.bbox[1], "x2": sample.bbox[2], "y2": sample.bbox[3]},
        "lower_court_intersection": sample.lower_court_intersection,
        "foot_court_score": sample.foot_court_score,
        "foot_points_on_court": sample.foot_points_on_court,
        "visible_landmarks": sample.visible_landmarks,
        "landmarks": sample.landmarks,
    }


def sample_to_response(sample: PlayerSample) -> dict[str, Any]:
    data = sample_to_json(sample)
    data["landmarks"] = [
        point
        for point in sample.landmarks
        if point["name"] in {"left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle"}
    ]
    return data


def court_group(video_dir: Path, video_path: Path) -> str:
    relative = video_path.relative_to(video_dir)
    if len(relative.parts) <= 1:
        return "_root"
    return "__".join(relative.parts[:-1])


def court_file(courts_dir: Path, video_dir: Path, video_path: Path) -> Path:
    return courts_dir / f"{court_group(video_dir, video_path)}.json"


def encode_png(cv2: Any, frame: Any) -> str:
    ok, buffer = cv2.imencode(".png", frame)
    if not ok:
        raise ValueError("Could not encode frame as PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def draw_bbox(cv2: Any, frame: Any, bbox: tuple[float, float, float, float], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_pose(cv2: Any, frame: Any, landmarks: list[dict[str, float | str]]) -> None:
    h, w = frame.shape[:2]
    for point in landmarks:
        if float(point["visibility"]) < 0.35:
            continue
        x = int(float(point["x"]) * w)
        y = int(float(point["y"]) * h)
        cv2.circle(frame, (x, y), 3, (40, 220, 240), -1)


HTML = r"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Action Pose Dataset Builder</title>
    <style>
      :root {
        --bg: #f6f7f8;
        --panel: #ffffff;
        --text: #202124;
        --muted: #667085;
        --line: #d7dce1;
        --accent: #1f7a5c;
        --accent-strong: #155e45;
        --danger: #b42318;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        background: var(--bg);
        color: var(--text);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 12px 18px;
        border-bottom: 1px solid var(--line);
        background: var(--panel);
      }
      h1 { margin: 0; font-size: 18px; line-height: 1.3; }
      main {
        display: grid;
        grid-template-columns: 340px minmax(0, 1fr);
        gap: 14px;
        padding: 14px;
      }
      aside, .stage {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
      }
      aside { padding: 14px; }
      section + section {
        margin-top: 14px;
        padding-top: 14px;
        border-top: 1px solid var(--line);
      }
      h2 { margin: 0 0 8px; font-size: 13px; }
      label {
        display: block;
        margin: 10px 0 6px;
        color: var(--muted);
        font-size: 12px;
      }
      select, input[type="text"] {
        width: 100%;
        min-height: 34px;
        border: 1px solid var(--line);
        border-radius: 6px;
        background: #fff;
        color: var(--text);
        font: inherit;
        font-size: 13px;
        padding: 7px 8px;
      }
      button {
        min-height: 34px;
        border: 1px solid var(--line);
        border-radius: 6px;
        background: #fff;
        color: var(--text);
        cursor: pointer;
        font: inherit;
        font-size: 13px;
        padding: 7px 10px;
      }
      button.primary {
        border-color: var(--accent);
        background: var(--accent);
        color: #fff;
      }
      button.primary:hover { background: var(--accent-strong); }
      button:disabled { cursor: not-allowed; opacity: .45; }
      .toolbar { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
      .stage { min-height: calc(100vh - 86px); padding: 12px; overflow: auto; }
      .canvas-wrap { width: 100%; overflow: auto; border-radius: 6px; background: #242424; }
      canvas { display: block; max-width: 100%; height: auto; cursor: crosshair; }
      .meta, .status, .points {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.45;
        white-space: pre-wrap;
      }
      .status.error { color: var(--danger); }
      .points { display: grid; gap: 3px; margin-top: 8px; }
      @media (max-width: 900px) {
        main { grid-template-columns: 1fr; }
        .stage { min-height: 460px; }
      }
    </style>
  </head>
  <body>
    <header><h1>Action Pose Dataset Builder</h1><div id="topStatus" class="meta"></div></header>
    <main>
      <aside>
        <section>
          <h2>Video</h2>
          <label for="videoSelect">Clip from data/raw_labeling</label>
          <select id="videoSelect"></select>
          <div class="toolbar">
            <button id="randomFrame" class="primary">Random Frame</button>
            <button id="loadCourt">Load Court</button>
          </div>
          <div id="videoMeta" class="meta"></div>
        </section>
        <section>
          <h2>Court Polygon</h2>
          <div class="toolbar">
            <button id="saveCourt">Save Court</button>
            <button id="undoPoint">Undo</button>
            <button id="clearPoints">Clear</button>
          </div>
          <div id="points" class="points"></div>
        </section>
        <section>
          <h2>Sample</h2>
          <label for="actionLabel">Action label</label>
          <input id="actionLabel" type="text" placeholder="receive, wait, set, attack..." />
          <div class="toolbar">
            <button id="processFrame" class="primary">Process Frame</button>
          </div>
          <div id="status" class="status"></div>
        </section>
      </aside>
      <div class="stage">
        <div class="canvas-wrap"><canvas id="canvas"></canvas></div>
      </div>
    </main>
    <script>
      const state = {
        videos: [],
        image: null,
        frameIndex: null,
        points: [],
        imagePng: null,
      };
      const videoSelect = document.getElementById("videoSelect");
      const canvas = document.getElementById("canvas");
      const ctx = canvas.getContext("2d");
      const statusEl = document.getElementById("status");
      const topStatus = document.getElementById("topStatus");
      const videoMeta = document.getElementById("videoMeta");
      const pointsEl = document.getElementById("points");

      async function requestJson(url, options = {}) {
        const response = await fetch(url, options);
        const data = await response.json();
        if (!response.ok || data.ok === false) throw new Error(data.error || response.statusText);
        return data;
      }

      async function loadVideos() {
        const data = await requestJson("/api/videos");
        state.videos = data.videos;
        videoSelect.innerHTML = "";
        for (const video of state.videos) {
          const option = document.createElement("option");
          option.value = video.relative_path;
          option.textContent = `${video.relative_path} (${video.size_mb} MB)`;
          videoSelect.appendChild(option);
        }
        topStatus.textContent = `${state.videos.length} videos`;
        updateDefaultActionLabel();
      }

      async function loadRandomFrame() {
        setStatus("Loading random frame...");
        updateDefaultActionLabel();
        const data = await requestJson(`/api/random-frame?video=${encodeURIComponent(videoSelect.value)}`);
        state.frameIndex = data.frame_index;
        state.imagePng = data.image_png;
        state.image = await loadImage(data.image_png);
        state.points = [];
        canvas.width = data.width;
        canvas.height = data.height;
        videoMeta.textContent = `${data.video_name}\nframe ${data.frame_index} / ${data.frame_count}\n${data.width}x${data.height}, ${data.fps.toFixed(2)} fps`;
        await loadCourt(false);
        draw();
        setStatus("Click the court boundary points, then process the frame.");
      }

      function loadImage(src) {
        return new Promise((resolve, reject) => {
          const image = new Image();
          image.onload = () => resolve(image);
          image.onerror = reject;
          image.src = src;
        });
      }

      async function loadCourt(showStatus = true) {
        if (!videoSelect.value) return;
        const data = await requestJson(`/api/court?video=${encodeURIComponent(videoSelect.value)}`);
        if (data.court_polygon && data.court_polygon.length) {
          state.points = data.court_polygon;
          draw();
          renderPoints();
          if (showStatus) setStatus("Loaded saved court polygon.");
        } else if (showStatus) {
          setStatus("No saved court polygon for this video.");
        }
      }

      async function saveCourt() {
        if (state.points.length < 3) return setStatus("Mark at least 3 points.", true);
        const data = await requestJson("/api/save-court", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({video: videoSelect.value, court_polygon: state.points}),
        });
        setStatus(`Saved court: ${data.court_file}`);
      }

      async function processFrame() {
        if (!state.image || state.frameIndex === null) return setStatus("Load a random frame first.", true);
        if (state.points.length < 3) return setStatus("Mark at least 3 court points.", true);
        updateDefaultActionLabel();
        setStatus("Running YOLO and MediaPipe...");
        const data = await requestJson("/api/process-frame", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            video: videoSelect.value,
            frame_index: state.frameIndex,
            court_polygon: state.points,
            action_label: document.getElementById("actionLabel").value || defaultActionLabel(),
          }),
        });
        state.image = await loadImage(data.annotated_png);
        draw(false);
        setStatus(`Saved ${data.saved_samples} samples.\nCSV: ${data.csv}\nJSONL: ${data.jsonl}`);
      }

      function draw(withOverlay = true) {
        if (!state.image) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(state.image, 0, 0);
        if (withOverlay) drawPolygon();
        renderPoints();
      }

      function drawPolygon() {
        if (!state.points.length) return;
        ctx.lineWidth = 4;
        ctx.strokeStyle = "#1f7a5c";
        ctx.fillStyle = "rgba(31, 122, 92, 0.18)";
        ctx.beginPath();
        const first = normalizePoint(state.points[0]);
        ctx.moveTo(first.x, first.y);
        for (const point of state.points.slice(1)) {
          const normalized = normalizePoint(point);
          ctx.lineTo(normalized.x, normalized.y);
        }
        if (state.points.length >= 3) ctx.closePath();
        ctx.fill();
        ctx.stroke();
        state.points.forEach((point, index) => {
          const normalized = normalizePoint(point);
          ctx.beginPath();
          ctx.arc(normalized.x, normalized.y, 7, 0, Math.PI * 2);
          ctx.fillStyle = "#2563b8";
          ctx.fill();
          ctx.fillStyle = "#fff";
          ctx.font = "12px sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(String(index + 1), normalized.x, normalized.y);
        });
      }

      function renderPoints() {
        pointsEl.innerHTML = "";
        state.points.forEach((point, index) => {
          const normalized = normalizePoint(point);
          const row = document.createElement("div");
          row.textContent = `${index + 1}: ${normalized.x.toFixed(1)}, ${normalized.y.toFixed(1)}`;
          pointsEl.appendChild(row);
        });
      }

      function normalizePoint(point) {
        if (Array.isArray(point)) return {x: Number(point[0]), y: Number(point[1])};
        return {x: Number(point.x), y: Number(point.y)};
      }

      function setStatus(message, error = false) {
        statusEl.textContent = message;
        statusEl.classList.toggle("error", error);
      }

      function defaultActionLabel() {
        const fileName = videoSelect.value.split("/").pop() || "";
        return fileName.replace(/\.[^.]+$/, "");
      }

      function updateDefaultActionLabel() {
        const input = document.getElementById("actionLabel");
        input.value = defaultActionLabel();
      }

      canvas.addEventListener("click", (event) => {
        if (!state.image) return;
        const rect = canvas.getBoundingClientRect();
        const sx = canvas.width / rect.width;
        const sy = canvas.height / rect.height;
        state.points.push({x: (event.clientX - rect.left) * sx, y: (event.clientY - rect.top) * sy});
        draw();
      });
      document.getElementById("randomFrame").addEventListener("click", () => loadRandomFrame().catch(err => setStatus(err.message, true)));
      document.getElementById("loadCourt").addEventListener("click", () => loadCourt(true).catch(err => setStatus(err.message, true)));
      document.getElementById("saveCourt").addEventListener("click", () => saveCourt().catch(err => setStatus(err.message, true)));
      document.getElementById("processFrame").addEventListener("click", () => processFrame().catch(err => setStatus(err.message, true)));
      document.getElementById("undoPoint").addEventListener("click", () => { state.points.pop(); draw(); });
      document.getElementById("clearPoints").addEventListener("click", () => { state.points = []; draw(); });
      videoSelect.addEventListener("change", () => { state.points = []; updateDefaultActionLabel(); draw(); });
      loadVideos().then(loadRandomFrame).catch(err => setStatus(err.message, true));
    </script>
  </body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
