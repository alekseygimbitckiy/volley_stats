from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from classify_rally_serve_receive import (  # noqa: E402
    PlayerBox,
    filter_player_boxes_by_team,
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


if __name__ == "__main__":
    unittest.main()
