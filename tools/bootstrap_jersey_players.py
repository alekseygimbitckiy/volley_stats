#!/usr/bin/env python3
"""Build player references automatically from OCR-readable shirt text."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from jersey_ocr import JerseyOCR, OCRIdentity, normalize_name, normalize_number
from player_roster import PlayerRoster, load_roster
from reid_osnet import OSNetEmbedder
from rebuild_player_embeddings_osnet import average_l2
from test_track_video import (
    extract_layout_polygon,
    filter_team_candidates,
    load_layout_data,
    suppress_duplicate_person_boxes,
    warn_layout_size_mismatch,
)


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample frames, OCR near-side shirt names/numbers, save confident crops, and build ReID embeddings."
    )
    parser.add_argument("video", help="Path to one video used for bootstrapping jersey players.")
    parser.add_argument("--layout", default="data/processed/calibrations/field_layout.json")
    parser.add_argument("--output", default="data/processed/auto_jersey_embeddings/player_embeddings.json")
    parser.add_argument("--snapshot-dir", default="data/processed/auto_jersey_embeddings/snapshots")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--device", default="cpu", help="Ultralytics device, for example cpu, 0, or cuda:0.")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-backend", choices=["imagenet-osnet", "soccernet-osnet"], default="soccernet-osnet")
    parser.add_argument("--checkpoint", default="external/soccernet-reid/osnet/model.osnet.pth.tar")
    parser.add_argument("--ocr-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ocr-backend", choices=["easyocr", "paddleocr"], default="easyocr")
    parser.add_argument("--ocr-languages", default="en", help="Comma-separated EasyOCR languages, for example en or en,ru.")
    parser.add_argument("--ocr-min-confidence", type=float, default=0.80)
    parser.add_argument("--identity-types", choices=["name-or-number", "name", "number"], default="name-or-number")
    parser.add_argument("--roster", default=None, help="JSON file with real player_id, jersey_number, and names/aliases.")
    parser.add_argument("--roster-name-threshold", type=float, default=0.78)
    parser.add_argument("--roster-name-margin", type=float, default=0.05)
    parser.add_argument("--min-name-length", type=int, default=3)
    parser.add_argument("--frames", type=int, default=100, help="Number of random frames to inspect.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--team-filter", choices=["court-nearest-6", "court-intersection", "largest"], default="court-nearest-6")
    parser.add_argument("--max-samples-per-identity", "--max-samples-per-jersey", dest="max_samples_per_identity", type=int, default=8)
    parser.add_argument("--min-person-confidence", type=float, default=0.25)
    parser.add_argument("--person-nms-iou", type=float, default=0.55)
    parser.add_argument("--fresh", action="store_true", help="Ignore samples already listed in the output JSON.")
    args = parser.parse_args()

    deps = import_dependencies()
    cv2 = deps["cv2"]

    video_path = resolve_project_path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    output_path = resolve_project_path(args.output)
    snapshot_dir = resolve_project_path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    layout_data = load_layout_data(resolve_project_path(args.layout))
    layout = extract_layout_polygon(layout_data)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0:
        raise SystemExit(f"Could not read frame count from {video_path}")

    warn_layout_size_mismatch(layout_data, width, height)
    frame_indices = sample_frame_indices(frame_count, args.frames, args.seed)

    print(f"Video: {video_path}")
    print(f"Frames: {frame_count}, sampled: {len(frame_indices)}, size: {width}x{height}")
    print(f"Loading YOLO: {args.yolo_model}")
    model = deps["YOLO"](args.yolo_model)
    roster = load_roster(args.roster, args.roster_name_threshold, args.roster_name_margin)
    if roster is not None:
        print(f"Loaded roster players: {len(roster.players)}")
    print(f"Loading jersey OCR on {args.ocr_device}")
    ocr = JerseyOCR(
        gpu=args.ocr_device == "cuda",
        min_confidence=args.ocr_min_confidence,
        languages=parse_languages(args.ocr_languages),
        backend=args.ocr_backend,
        model_dir=f"external/{args.ocr_backend}",
    )

    data = load_existing_output(output_path, fresh=args.fresh)
    samples = data["samples"]
    existing_counts = count_samples_by_player(samples)
    next_sample_number = next_sample_index(samples)
    found_this_run = 0

    for offset, frame_idx in enumerate(frame_indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        person_candidates = detect_person_candidates(model, frame, args.device, args.min_person_confidence)
        person_candidates = suppress_duplicate_person_boxes(person_candidates, args.person_nms_iou)
        near_players = filter_team_candidates(person_candidates, layout, width, height, mode=args.team_filter)
        for box in near_players[:6]:
            x1, y1, x2, y2, person_confidence = box
            crop = crop_bbox(frame, box)
            if crop is None:
                continue
            if roster is not None:
                roster_result = read_roster_crop_identity(ocr, cv2, crop, args, roster)
                if roster_result is None:
                    continue
                identity, roster_match, number_identity = roster_result
            else:
                identity = read_crop_identity(ocr, cv2, crop, args)
                if identity is None:
                    continue
                roster_match = None
                number_identity = identity if identity.kind == "number" else None
            player_id = roster_match.player.player_id if roster_match is not None else identity_to_player_id(identity.kind, identity.value)
            if existing_counts.get(player_id, 0) >= args.max_samples_per_identity:
                continue

            player_dir = snapshot_dir / player_id
            player_dir.mkdir(parents=True, exist_ok=True)
            sample_id = f"sample_{next_sample_number:04d}"
            file_name = f"{sample_id}_{video_path.stem}_frame_{frame_idx}_{player_id}.png"
            snapshot_path = player_dir / file_name
            cv2.imwrite(str(snapshot_path), crop)
            number = (
                roster_match.player.jersey_number
                if roster_match is not None
                else identity.value if identity.kind == "number" else None
            )
            shirt_name = (
                roster_match.player.names[0]
                if roster_match is not None and roster_match.player.names
                else identity.value if identity.kind == "name" else None
            )

            samples.append(
                {
                    "sample_id": sample_id,
                    "player_id": player_id,
                    "source_video": str(video_path),
                    "frame": frame_idx,
                    "time_ms": int(round(frame_idx * 1000.0 / fps)),
                    "bbox": {
                        "x1": round(float(x1), 3),
                        "y1": round(float(y1), 3),
                        "x2": round(float(x2), 3),
                        "y2": round(float(y2), 3),
                    },
                    "person_confidence": float(person_confidence),
                    "ocr_kind": identity.kind,
                    "ocr_text": identity.value,
                    "ocr_raw_text": identity.raw_text,
                    "ocr_confidence": float(identity.confidence),
                    "ocr_number_text": number_identity.value if number_identity is not None else None,
                    "ocr_number_confidence": float(number_identity.confidence) if number_identity is not None else None,
                    "roster_match_kind": roster_match.kind if roster_match is not None else None,
                    "roster_match_score": roster_match.score if roster_match is not None else None,
                    "jersey_number": number,
                    "jersey_number_confidence": float(number_identity.confidence) if number_identity is not None and number else None,
                    "shirt_name": shirt_name,
                    "shirt_name_confidence": float(identity.confidence) if identity.kind == "name" and shirt_name else None,
                    "snapshot_path": str(snapshot_path),
                }
            )
            existing_counts[player_id] = existing_counts.get(player_id, 0) + 1
            next_sample_number += 1
            found_this_run += 1
            print(
                f"  frame {frame_idx}: {player_id} {identity.kind}={identity.value} "
                f"ocr={identity.confidence:.2f} saved={snapshot_path.name}"
            )
        if offset % 10 == 0:
            print(f"Processed sampled frames: {offset}/{len(frame_indices)}")

    cap.release()
    if not samples:
        raise SystemExit("No OCR jersey snapshots were saved.")

    print(f"New OCR snapshots: {found_this_run}")
    build_embeddings(data, samples, args, deps)
    data["metadata"] = {
        "source": "bootstrap_jersey_players.py",
        "video": str(video_path),
        "sampled_frames": len(frame_indices),
        "team_filter": args.team_filter,
        "roster": str(resolve_project_path(args.roster)) if args.roster else None,
    }
    data["saved_files"] = {
        "embeddings": str(output_path),
        "snapshots": str(snapshot_dir),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved embeddings: {output_path}")
    print(f"Saved snapshots: {snapshot_dir}")
    return 0


def import_dependencies() -> dict[str, Any]:
    try:
        import cv2  # type: ignore
        from ultralytics import YOLO  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing package: {exc.name}") from exc
    return {"cv2": cv2, "YOLO": YOLO}


def parse_languages(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()] or ["en"]


def sample_frame_indices(frame_count: int, sample_count: int, seed: int) -> list[int]:
    count = min(frame_count, max(1, sample_count))
    rng = random.Random(seed)
    return sorted(rng.sample(range(frame_count), count))


def detect_person_candidates(model: Any, frame: Any, device: str, min_confidence: float) -> list[tuple[float, float, float, float, float]]:
    result = model.predict(frame, verbose=False, conf=min_confidence, device=device)[0]
    candidates = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:
            continue
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        confidence = float(box.conf[0])
        candidates.append((x1, y1, x2, y2, confidence))
    return candidates


def read_crop_identity(ocr: JerseyOCR, cv2: Any, crop: Any, args: argparse.Namespace) -> Any | None:
    if args.identity_types == "name":
        name, confidence = ocr.read_name(cv2, crop, min_length=args.min_name_length)
        if not name or confidence is None:
            return None
        return SimpleIdentity("name", name, confidence, name)
    if args.identity_types == "number":
        number, confidence = ocr.read_number(cv2, crop)
        number = normalize_number(number or "")
        if not number or confidence is None:
            return None
        return SimpleIdentity("number", number, confidence, number)
    return ocr.read_identity(cv2, crop, prefer_name=True, min_name_length=args.min_name_length)


def read_roster_crop_identity(
    ocr: JerseyOCR,
    cv2: Any,
    crop: Any,
    args: argparse.Namespace,
    roster: PlayerRoster,
) -> tuple[OCRIdentity, Any, OCRIdentity | None] | None:
    name_identity = None
    number_identity = None

    if args.identity_types in ("name", "name-or-number"):
        name, confidence = ocr.read_name(cv2, crop, min_length=args.min_name_length)
        if name and confidence is not None:
            name_identity = OCRIdentity("name", name, float(confidence), name)

    if args.identity_types in ("number", "name-or-number"):
        number, confidence = ocr.read_number(cv2, crop)
        number = normalize_number(number or "")
        if number and confidence is not None:
            number_identity = OCRIdentity("number", number, float(confidence), number)

    if name_identity is not None:
        name_match = roster.match_identity(name_identity)
        if name_match is None:
            return None
        roster_number = name_match.player.jersey_number
        if roster_number:
            if number_identity is None or normalize_number(number_identity.value) != roster_number:
                return None
        return name_identity, name_match, number_identity

    if number_identity is not None:
        number_match = roster.match_identity(number_identity)
        if number_match is not None:
            return number_identity, number_match, number_identity

    return None


class SimpleIdentity:
    def __init__(self, kind: str, value: str, confidence: float, raw_text: str) -> None:
        self.kind = kind
        self.value = value
        self.confidence = confidence
        self.raw_text = raw_text


def identity_to_player_id(kind: str, value: str) -> str:
    if kind == "name":
        return f"name_{safe_token(value)}"
    return f"jersey_{safe_token(value)}"


def safe_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned or "unknown"


def crop_bbox(frame: Any, box: tuple[float, float, float, float, float]) -> Any | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2, _confidence = box
    x1i = max(0, min(width - 1, int(round(x1))))
    y1i = max(0, min(height - 1, int(round(y1))))
    x2i = max(0, min(width, int(round(x2))))
    y2i = max(0, min(height, int(round(y2))))
    if x2i <= x1i or y2i <= y1i:
        return None
    crop = frame[y1i:y2i, x1i:x2i]
    return crop if crop.size else None


def load_existing_output(path: Path, fresh: bool) -> dict[str, Any]:
    if path.exists() and not fresh:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("players", [])
        data.setdefault("samples", [])
        return data
    return {"schema_version": 1, "players": [], "samples": []}


def count_samples_by_player(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        player_id = sample.get("player_id")
        if player_id:
            counts[player_id] = counts.get(player_id, 0) + 1
    return counts


def next_sample_index(samples: list[dict[str, Any]]) -> int:
    max_index = 0
    for sample in samples:
        sample_id = str(sample.get("sample_id", ""))
        if sample_id.startswith("sample_"):
            try:
                max_index = max(max_index, int(sample_id.split("_", 1)[1]))
            except ValueError:
                pass
    return max_index + 1


def build_embeddings(data: dict[str, Any], samples: list[dict[str, Any]], args: argparse.Namespace, deps: dict[str, Any]) -> None:
    cv2 = deps["cv2"]
    checkpoint = args.checkpoint if args.embedding_backend == "soccernet-osnet" else None
    print(f"Loading ReID backend: {args.embedding_backend} on {args.embedding_device}")
    embedder = OSNetEmbedder(
        device=args.embedding_device,
        checkpoint_path=checkpoint,
        pretrained=args.embedding_backend == "imagenet-osnet",
    )

    grouped: dict[str, list[list[float]]] = {}
    grouped_numbers: dict[str, list[tuple[str, float]]] = {}
    grouped_names: dict[str, list[tuple[str, float]]] = {}
    valid_samples = []
    for sample in samples:
        player_id = sample.get("player_id")
        snapshot_path = sample.get("snapshot_path")
        if not player_id or not snapshot_path:
            continue
        image = cv2.imread(str(resolve_project_path(snapshot_path)))
        if image is None:
            print(f"  warning: could not read snapshot {snapshot_path}")
            continue
        embedding = embedder.embed_bgr(cv2, image)
        sample["embedding"] = embedding
        grouped.setdefault(player_id, []).append(embedding)
        number = normalize_number(sample.get("jersey_number") or (player_id.replace("jersey_", "") if player_id.startswith("jersey_") else ""))
        if number:
            grouped_numbers.setdefault(player_id, []).append((number, float(sample.get("jersey_number_confidence") or 0.0)))
        name = normalize_name(sample.get("shirt_name") or (player_id.replace("name_", "") if player_id.startswith("name_") else ""))
        if name:
            grouped_names.setdefault(player_id, []).append((name, float(sample.get("shirt_name_confidence") or sample.get("ocr_confidence") or 0.0)))
        valid_samples.append(sample)

    players = []
    for player_id in sorted(grouped):
        jersey_number, jersey_confidence = best_jersey_number(grouped_numbers.get(player_id, []))
        shirt_name, shirt_name_confidence = best_jersey_number(grouped_names.get(player_id, []))
        player_samples = [sample for sample in valid_samples if sample.get("player_id") == player_id]
        players.append(
            {
                "player_id": player_id,
                "jersey_number": jersey_number,
                "jersey_number_confidence": jersey_confidence,
                "shirt_name": shirt_name,
                "shirt_name_confidence": shirt_name_confidence,
                "embedding": average_l2(grouped[player_id]),
                "sample_count": len(player_samples),
                "sample_ids": [sample.get("sample_id") for sample in player_samples],
            }
        )

    data["players"] = players
    data["samples"] = valid_samples
    data["embedding"] = {
        "type": f"torchreid_osnet_x1_0_{args.embedding_backend}_v1",
        "distance": "euclidean after L2 normalization",
        "input_size": {"width": 128, "height": 256},
        "checkpoint": str(resolve_project_path(args.checkpoint)) if checkpoint else None,
        "ocr": {
            "type": "easyocr_shirt_text_v1",
            "preferred": True,
            "fallback": "Use OSNet embedding when OCR cannot parse a confident unique shirt name or jersey number.",
            "min_confidence": args.ocr_min_confidence,
            "identity_types": args.identity_types,
        },
        "notes": "Automatically bootstrapped from confident OCR-readable shirt crops.",
    }


def best_jersey_number(numbers: list[tuple[str, float]]) -> tuple[str | None, float | None]:
    if not numbers:
        return None, None
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    for number, confidence in numbers:
        scores[number] = scores.get(number, 0.0) + confidence
        counts[number] = counts.get(number, 0) + 1
    best = max(scores, key=lambda number: (counts[number], scores[number]))
    return best, scores[best] / max(1, counts[best])


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
