#!/usr/bin/env python3
"""Rebuild saved player embeddings from snapshot crops using Torchreid OSNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reid_osnet import OSNetEmbedder


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute OSNet embeddings for stored player snapshots.")
    parser.add_argument("--input", default="data/processed/player_embeddings/player_embeddings.json")
    parser.add_argument("--output", default="data/processed/player_embeddings/player_embeddings.json")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or 0.")
    parser.add_argument("--model-name", default="osnet_x1_0")
    args = parser.parse_args()

    deps = import_dependencies()
    cv2 = deps["cv2"]

    input_path = resolve_project_path(args.input)
    output_path = resolve_project_path(args.output)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    samples = data.get("samples", [])
    if not samples:
        raise SystemExit(f"No samples found in {input_path}")

    print(f"Loading OSNet: {args.model_name} on {args.device}")
    embedder = OSNetEmbedder(device=args.device, model_name=args.model_name)

    sample_embeddings: dict[str, list[float]] = {}
    grouped: dict[str, list[list[float]]] = {}
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

    data["embedding"] = {
        "type": f"torchreid_{args.model_name}_imagenet_v1",
        "distance": "euclidean after L2 normalization",
        "input_size": {"width": 128, "height": 256},
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
