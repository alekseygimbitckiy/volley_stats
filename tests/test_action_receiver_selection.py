from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from classify_rally_serve_receive import (  # noqa: E402
    PlayerBox,
    evaluate_reception_quality,
    filter_player_boxes_by_team,
    find_pass_origin_anchor,
    find_receiver,
)


class ActionReceiverSelectionTests(unittest.TestCase):
    def test_plausible_known_player_beats_closer_unknown_prediction(self) -> None:
        boxes = {
            40: [
                PlayerBox(40, "known", "jersey_9", (60, 100, 100, 220), 0.9, "near", True),
                PlayerBox(40, "ghost", None, (80, 45, 120, 125), 0.4, "near", False),
            ]
        }
        receiver = find_receiver(
            boxes,
            ball_x=100,
            ball_y=50,
            reception_frame=40,
            frame_window=0,
            max_distance=180,
            dispute_margin=35,
            known_player_bonus=110,
            prediction_penalty=35,
        )
        self.assertIsNotNone(receiver)
        self.assertEqual(receiver["player_id"], "jersey_9")

    def test_classified_near_removes_opponents_and_keeps_unknown_near_tracks(self) -> None:
        boxes = {
            0: [
                PlayerBox(0, "near", None, (0, 0, 10, 20), 0.9, "near"),
                PlayerBox(0, "far", None, (20, 0, 30, 20), 0.9, "opponent"),
            ]
        }
        filtered = filter_player_boxes_by_team(boxes, [], "classified-near")
        self.assertEqual([box.track_id for box in filtered[0]], ["near"])

    def test_pass_origin_uses_maximum_observed_y_before_pass(self) -> None:
        tracks = {
            "setter": [
                PlayerBox(93, "setter", "jersey_8", (20, 0, 40, 500), 0.9, "near", True),
                PlayerBox(94, "setter", "jersey_8", (20, 0, 40, 300), 0.9, "near", True),
                PlayerBox(95, "setter", "jersey_8", (20, 0, 40, 350), 0.9, "near", False),
                PlayerBox(98, "setter", "jersey_8", (30, 0, 50, 280), 0.9, "near", True),
                PlayerBox(102, "setter", "jersey_8", (30, 0, 50, 600), 0.9, "near", True),
            ]
        }
        origin = find_pass_origin_anchor(
            passer={"track_id": "setter", "player_id": "jersey_8"},
            player_tracks_by_id=tracks,
            pass_frame=100,
            lookback_frames=6,
        )
        self.assertIsNotNone(origin)
        self.assertEqual(origin["frame"], 94)
        self.assertEqual((origin["x"], origin["y"]), (30.0, 300))
        self.assertEqual(origin["method"], "max_y_same_track_pre_pass")

    def test_reception_score_uses_prepass_origin_instead_of_airborne_box(self) -> None:
        actions = [
            {
                "frame": 2,
                "receiver": {
                    "player_id": "receiver",
                    "pose_action": {"probabilities": {"recive_top": 0.9}},
                },
            },
            {
                "frame": 10,
                "vy_before_px_per_frame": 4.0,
                "vy_after_px_per_frame": -3.0,
                "receiver": {
                    "track_id": "setter",
                    "player_id": "jersey_8",
                    "frame": 10,
                    "bbox": {"x1": 40, "y1": 0, "x2": 60, "y2": 50},
                },
            },
        ]
        tracks = {
            "setter": [
                PlayerBox(8, "setter", "jersey_8", (40, 0, 60, 90), 0.9, "near", True),
                PlayerBox(9, "setter", "jersey_8", (40, 0, 60, 70), 0.9, "near", True),
                PlayerBox(10, "setter", "jersey_8", (40, 0, 60, 50), 0.9, "near", True),
            ]
        }
        zones = [{"zone_id": "perfect", "label": "perfect", "score": 1, "polygon": [(30, 80), (70, 80), (70, 100), (30, 100)]}]
        result = evaluate_reception_quality(
            actions,
            zones,
            receive_probability_threshold=0.33,
            player_tracks_by_id=tracks,
            pass_origin_lookback_frames=2,
        )
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["pass"]["origin_court_anchor"]["frame"], 8)
        self.assertEqual(result["pass"]["court_anchor"], {"x": 50.0, "y": 90})
        self.assertEqual(result["pass"]["pass_time_court_anchor"], {"x": 50.0, "y": 50.0})


if __name__ == "__main__":
    unittest.main()
