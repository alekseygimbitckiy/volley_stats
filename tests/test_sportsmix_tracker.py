from dataclasses import dataclass
import unittest

import numpy as np

from tools.sportsmix_tracker import SportsMixTracker


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    embedding: list[float]
    confidence: float = 0.9
    overlap_iou: float = 0.0


class SportsMixTrackerTests(unittest.TestCase):
    def test_appearance_keeps_ids_when_players_cross(self) -> None:
        tracker = SportsMixTracker(
            max_age=3,
            motion_weight=0.35,
            appearance_weight=0.60,
            center_weight=0.05,
            match_threshold=0.95,
        )
        first = tracker.update(
            [Detection((0, 0, 20, 40), [1, 0]), Detection((80, 0, 100, 40), [0, 1])],
            0,
        )
        self.assertEqual([item.track_id for item in first], [1, 2])

        tracker.update(
            [Detection((20, 0, 40, 40), [1, 0]), Detection((60, 0, 80, 40), [0, 1])],
            1,
        )
        crossed = tracker.update(
            [Detection((50, 0, 70, 40), [1, 0]), Detection((30, 0, 50, 40), [0, 1])],
            2,
        )
        by_source = {item.source_index: item.track_id for item in crossed if item.observed}
        self.assertEqual(by_source, {0: 1, 1: 2})

    def test_overlap_crop_does_not_contaminate_template(self) -> None:
        tracker = SportsMixTracker(max_age=2, overlap_iou=0.25)
        tracker.update([Detection((0, 0, 20, 40), [1, 0])], 0)
        tracker.update([Detection((1, 0, 21, 40), [0, 1], overlap_iou=0.8)], 1)
        np.testing.assert_allclose(tracker._states[0].embedding, np.asarray([1.0, 0.0]))

    def test_unmatched_track_is_predicted_for_configured_age(self) -> None:
        tracker = SportsMixTracker(max_age=2)
        tracker.update([Detection((0, 0, 20, 40), [1, 0])], 0)
        one = tracker.update([], 1)
        two = tracker.update([], 2)
        three = tracker.update([], 3)
        self.assertEqual(len(one), 1)
        self.assertEqual(len(two), 1)
        self.assertEqual(one[0].track_id, 1)
        self.assertEqual(two[0].track_id, 1)
        self.assertFalse(one[0].observed)
        self.assertFalse(two[0].observed)
        self.assertEqual(three, [])


if __name__ == "__main__":
    unittest.main()
