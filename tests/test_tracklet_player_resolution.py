import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from test_track_video import (  # noqa: E402
    PlayerDetection,
    PlayerTrack,
    classify_track_teams,
    interpolate_internal_track_gaps,
    resolve_tracklet_identities,
    split_tracklets_on_appearance_change,
)


def detection(
    frame: int,
    *,
    player_id: str | None = None,
    source: str | None = None,
    jersey_confidence: float | None = None,
    embedding: list[float] | None = None,
    color: tuple[float, float, float] = (40, 100, 90),
) -> PlayerDetection:
    return PlayerDetection(
        frame=frame,
        bbox=(10 + frame, 10, 30 + frame, 50),
        confidence=0.9,
        embedding=embedding or [],
        player_id=player_id,
        player_distance=None,
        jersey_number="9" if player_id else None,
        jersey_confidence=jersey_confidence,
        identity_source=source,
        uniform_color=color,
    )


class TrackletPlayerResolutionTests(unittest.TestCase):
    def test_future_ocr_labels_the_entire_earlier_tracklet(self) -> None:
        track = PlayerTrack(
            track_id=1,
            team="near",
            detections=[
                detection(0, embedding=[1, 0]),
                detection(10, player_id="jersey_9", source="ocr_number", jersey_confidence=0.95, embedding=[1, 0]),
            ],
        )
        resolve_tracklet_identities(
            [track], None, {"jersey_9": [[1, 0]]},
            reid_threshold=0.9, reid_margin=0.1, reid_min_confidence=0.1, ocr_min_confidence=0.85,
        )
        self.assertEqual({item.player_id for item in track.detections}, {"jersey_9"})
        self.assertEqual(track.identity_source, "tracklet_ocr_future")

    def test_ambiguous_tracklet_remains_unknown(self) -> None:
        track = PlayerTrack(track_id=1, team="near", detections=[detection(0, embedding=[1, 0])])
        resolve_tracklet_identities(
            [track], None, {"jersey_8": [[1, 0]], "jersey_9": [[0.999, 0.001]]},
            reid_threshold=0.9, reid_margin=0.1, reid_min_confidence=0.1, ocr_min_confidence=0.85,
        )
        self.assertIsNone(track.resolved_player_id)
        self.assertIsNone(track.detections[0].player_id)

    def test_opponent_track_cannot_keep_roster_identity(self) -> None:
        track = PlayerTrack(
            track_id=1,
            team="opponent",
            detections=[detection(0, player_id="jersey_8", source="ocr_number", jersey_confidence=0.99)],
        )
        resolve_tracklet_identities(
            [track], None, {}, reid_threshold=0.9, reid_margin=0.1, reid_min_confidence=0.1, ocr_min_confidence=0.85,
        )
        self.assertIsNone(track.detections[0].player_id)
        self.assertEqual(track.identity_source, "opponent_team")

    def test_team_is_classified_after_tracking(self) -> None:
        near = PlayerTrack(track_id=1, detections=[detection(0, color=(40, 100, 90))])
        opponent = PlayerTrack(track_id=2, detections=[detection(0, color=(100, 200, 200))])
        polygon = [(0, 100), (100, 100), (100, 0), (0, 0)]
        classify_track_teams(
            [near, opponent], polygon, [(40, 100, 90)], {}, np, color_threshold=20, color_margin=10,
            reid_threshold=0.9, reid_margin=0.1, reid_min_confidence=0.1,
            dark_value_threshold=45, dark_reid_confidence=0.3,
        )
        self.assertEqual(near.team, "near")
        self.assertEqual(opponent.team, "opponent")

    def test_short_internal_gap_is_interpolated(self) -> None:
        track = PlayerTrack(track_id=1, detections=[detection(0), detection(3)])
        interpolate_internal_track_gaps([track], max_gap=2)
        self.assertEqual([item.frame for item in track.detections], [0, 1, 2, 3])
        self.assertFalse(track.detections[1].observed)

    def test_sustained_appearance_switch_splits_tracklet(self) -> None:
        track = PlayerTrack(
            track_id=4,
            detections=[
                *[detection(frame, color=(75, 110, 60)) for frame in range(6)],
                *[detection(frame, color=(25, 70, 25)) for frame in range(6, 12)],
            ],
        )
        split = split_tracklets_on_appearance_change(
            [track], np, threshold=60, min_observations=4,
        )
        self.assertEqual([item.track_id for item in split], ["4.a", "4.b"])
        self.assertEqual(split[0].hard_end_frame, 5)


if __name__ == "__main__":
    unittest.main()
