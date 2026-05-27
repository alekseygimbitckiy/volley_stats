#!/usr/bin/env python3
"""Rebuild saved player embeddings from snapshot crops using Torchreid OSNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jersey_ocr import JerseyOCR
from reid_osnet import OSNetEmbedder


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute OSNet embeddings for stored player snapshots.")
    parser.add_argument("--input", default="data/processed/player_embeddings/player_embeddings.json")
    parser.add_argument("--output", default="data/processed/player_embeddings/player_embeddings.json")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or 0.")
    parser.add_argument("--model-name", default="osnet_x1_0")
    parser.add_argument(
        "--backend",
        choices=["imagenet-osnet", "soccernet-osnet"],
        default="imagenet-osnet",
        help="Which ReID weights to use.",
    )
    parser.add_argument(
        "--checkpoint",
        default="external/soccernet-reid/osnet/model.osnet.pth.tar",
        help="SoccerNet-ReID OSNet checkpoint path.",
    )
    parser.add_argument("--ocr", choices=["auto", "off"], default="auto", help="Read jersey numbers from snapshots when possible.")
    parser.add_argument("--ocr-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ocr-backend", choices=["easyocr", "paddleocr"], default="easyocr")
    parser.add_argument("--ocr-min-confidence", type=float, default=0.35)
    args = parser.parse_args()

    deps = import_dependencies()
    cv2 = deps["cv2"]

    input_path = resolve_project_path(args.input)
    output_path = resolve_project_path(args.output)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    samples = data.get("samples", [])
    if not samples:
        raise SystemExit(f"No samples found in {input_path}")

    checkpoint = args.checkpoint if args.backend == "soccernet-osnet" else None
    print(f"Loading ReID backend: {args.backend} ({args.model_name}) on {args.device}")
    embedder = OSNetEmbedder(
        device=args.device,
        model_name=args.model_name,
        checkpoint_path=checkpoint,
        pretrained=args.backend == "imagenet-osnet",
    )
    ocr = create_ocr(args)

    sample_embeddings: dict[str, list[float]] = {}
    grouped: dict[str, list[list[float]]] = {}
    grouped_numbers: dict[str, list[tuple[str, float]]] = {}
    for sample in samples:
        sample_id = sample.get("sample_id")
        player_id = sample.get("player_id")
        snapshot_path = sample.get("snapshot_path")
        if not sample_id or not player_id or not snapshot_path:
            continue
        image_path = resolve_project_path(snapshot_path)
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  warning: could not read {image_path}")
            continue
        embedding = embedder.embed_bgr(cv2, image)
        sample["embedding"] = embedding
        sample_embeddings[sample_id] = embedding
        grouped.setdefault(player_id, []).append(embedding)
        if ocr is not None:
            number, number_confidence = ocr.read_number(cv2, image)
            sample["jersey_number"] = number
            sample["jersey_number_confidence"] = number_confidence
            if number:
                grouped_numbers.setdefault(player_id, []).append((number, float(number_confidence or 0.0)))
                print(
                    f"  {player_id} {sample_id}: {image_path.name} "
                    f"jersey={number} conf={float(number_confidence or 0.0):.2f}"
                )
                continue
        print(f"  {player_id} {sample_id}: {image_path.name}")

    if not grouped:
        raise SystemExit("No embeddings were computed.")

    for player in data.get("players", []):
        player_id = player.get("player_id")
        embeddings = grouped.get(player_id, [])
        if not embeddings:
            continue
        player["embedding"] = average_l2(embeddings)
        player["sample_count"] = len(embeddings)
        player["sample_ids"] = [
            sample.get("sample_id")
            for sample in samples
            if sample.get("player_id") == player_id and sample.get("sample_id") in sample_embeddings
        ]
        jersey_number, jersey_confidence = best_jersey_number(grouped_numbers.get(player_id, []))
        player["jersey_number"] = jersey_number
        player["jersey_number_confidence"] = jersey_confidence

    data["embedding"] = {
        "type": f"torchreid_{args.model_name}_{args.backend}_v1",
        "distance": "euclidean after L2 normalization",
        "input_size": {"width": 128, "height": 256},
        "checkpoint": str(resolve_project_path(args.checkpoint)) if checkpoint else None,
        "ocr": {
            "type": "easyocr_digits_v1" if ocr is not None else None,
            "preferred": ocr is not None,
            "fallback": "Use OSNet embedding when OCR cannot parse a jersey number.",
            "min_confidence": args.ocr_min_confidence,
        },
        "notes": "Player references are averaged OSNet embeddings from manually selected snapshot crops.",
    }
    data.setdefault("saved_files", {})["embeddings"] = str(output_path)

    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved OSNet embeddings: {output_path}")
    return 0


def import_dependencies() -> dict[str, Any]:
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing package: opencv-python") from exc
    return {"cv2": cv2}


def create_ocr(args: argparse.Namespace) -> JerseyOCR | None:
    if args.ocr == "off":
        return None
    try:
        print(f"Loading jersey OCR on {args.ocr_device}")
        return JerseyOCR(
            gpu=args.ocr_device == "cuda",
            min_confidence=args.ocr_min_confidence,
            backend=args.ocr_backend,
            model_dir=f"external/{args.ocr_backend}",
        )
    except Exception as exc:  # noqa: BLE001 - OCR is optional fallback metadata.
        print(f"  warning: OCR unavailable, continuing with OSNet embeddings only: {exc}")
        return None


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


def average_l2(vectors: list[list[float]]) -> list[float]:
    length = len(vectors[0])
    avg = [0.0] * length
    for vector in vectors:
        for idx, value in enumerate(vector):
            avg[idx] += float(value)
    avg = [value / len(vectors) for value in avg]
    norm = sum(value * value for value in avg) ** 0.5
    if norm <= 1e-12:
        return avg
    return [value / norm for value in avg]


if __name__ == "__main__":
    raise SystemExit(main())
