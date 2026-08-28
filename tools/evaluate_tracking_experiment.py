#!/usr/bin/env python3
"""Score tracking ablations at known volleyball contact frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tracking_json", nargs="+", help="Tracking JSON paths, optionally NAME=PATH.")
    parser.add_argument(
        "--config",
        default="tests/fixtures/volleydzen_test_tracking_eval.json",
        help="Ground-truth action-frame boxes and identities.",
    )
    parser.add_argument("--min-iou", type=float, default=0.20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    reports = []
    for value in args.tracking_json:
        if "=" in value:
            name, raw_path = value.split("=", 1)
        else:
            raw_path = value
            name = Path(raw_path).parent.name
        reports.append(evaluate(name, Path(raw_path), config, args.min_iou))

    for report in reports:
        print(
            f"{report['experiment']}: identity={report['correct_identities']}/{report['target_count']} "
            f"coverage={report['covered_targets']}/{report['target_count']} "
            f"red_opponents={report['red_opponents']}/{report['opponent_target_count']} "
            f"opponent_label_leaks={report['opponent_label_leaks']}"
        )
        for target in report["targets"]:
            print(
                f"  {target['action']} f={target['frame']}: expected={target['expected_player_id']} "
                f"predicted={target['predicted_player_id']} team={target['team']} "
                f"track={target['track_id']} iou={target['iou']:.3f} "
                f"{'PASS' if target['correct'] else 'FAIL'}"
            )
        for target in report["opponent_targets"]:
            print(
                f"  opponent f={target['frame']}: team={target['team']} track={target['track_id']} "
                f"iou={target['iou']:.3f} {'PASS' if target['correct'] else 'FAIL'}"
            )
    if args.output:
        Path(args.output).write_text(json.dumps({"experiments": reports}, indent=2) + "\n", encoding="utf-8")
    return 0


def evaluate(name: str, path: Path, config: dict[str, Any], min_iou: float) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = {
        int(row.get("frame", 0)): row.get("detections", [])
        for row in data.get("visible_player_detections", [])
    }
    results = []
    for target in config.get("targets", []):
        target_bbox = bbox_tuple(target["bbox"])
        candidates = frames.get(int(target["frame"]), [])
        ranked = sorted(
            ((bbox_iou(target_bbox, bbox_tuple(item.get("bbox") or {})), item) for item in candidates),
            key=lambda pair: pair[0],
            reverse=True,
        )
        overlap, best = ranked[0] if ranked else (0.0, {})
        covered = overlap >= min_iou
        predicted = (best.get("player_id") or best.get("raw_player_id")) if covered else None
        team = str(best.get("team") or "unknown") if covered else "missing"
        results.append(
            {
                "action": target["action"],
                "frame": int(target["frame"]),
                "expected_player_id": target["expected_player_id"],
                "predicted_player_id": predicted,
                "team": team,
                "track_id": best.get("track_id") if covered else None,
                "iou": float(overlap),
                "covered": covered,
                "correct": covered and team != "opponent" and predicted == target["expected_player_id"],
            }
        )

    opponent_results = []
    for target in config.get("opponent_targets", []):
        target_bbox = bbox_tuple(target["bbox"])
        candidates = frames.get(int(target["frame"]), [])
        ranked = sorted(
            ((bbox_iou(target_bbox, bbox_tuple(item.get("bbox") or {})), item) for item in candidates),
            key=lambda pair: pair[0],
            reverse=True,
        )
        overlap, best = ranked[0] if ranked else (0.0, {})
        covered = overlap >= min_iou
        team = str(best.get("team") or "unknown") if covered else "missing"
        opponent_results.append(
            {
                "frame": int(target["frame"]),
                "team": team,
                "track_id": best.get("track_id") if covered else None,
                "iou": float(overlap),
                "covered": covered,
                "correct": covered and team == "opponent",
            }
        )

    opponent_label_leaks = 0
    for row in frames.values():
        for detection in row:
            if detection.get("team") == "opponent" and (
                detection.get("player_id") or detection.get("raw_player_id")
            ):
                opponent_label_leaks += 1
    return {
        "experiment": name,
        "tracking_json": str(path),
        "target_count": len(results),
        "covered_targets": sum(result["covered"] for result in results),
        "correct_identities": sum(result["correct"] for result in results),
        "opponent_target_count": len(opponent_results),
        "red_opponents": sum(result["correct"] for result in opponent_results),
        "opponent_label_leaks": opponent_label_leaks,
        "targets": results,
        "opponent_targets": opponent_results,
    }


def bbox_tuple(value: dict[str, Any]) -> tuple[float, float, float, float]:
    return tuple(float(value.get(key, 0.0)) for key in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


if __name__ == "__main__":
    raise SystemExit(main())
