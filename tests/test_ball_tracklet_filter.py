import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from test_track_video import BallObservation, filter_offline_ball_tracklets  # noqa: E402


def ball(frame: int, x: float, y: float) -> BallObservation:
    return BallObservation(frame, x, y, 5.0, 1.0, "test")


class OfflineBallTrackletFilterTests(unittest.TestCase):
    def test_removes_short_static_prefix_and_keeps_rally_segments(self) -> None:
        false_prefix = [ball(frame, 100 + frame % 2, 200 + frame) for frame in range(0, 7)]
        serve_flight = [ball(frame, 300 + (frame - 13) * 12, 250 - (frame - 13) * 3) for frame in range(13, 31)]
        false_middle = [ball(frame, 900, 400 + frame % 2) for frame in range(42, 48)]
        rally = [ball(frame, 500 + (frame - 60) * 6, 300 + (frame - 60) * 2) for frame in range(60, 81)]

        filtered, report = filter_offline_ball_tracklets(
            false_prefix + serve_flight + false_middle + rally,
            frame_count=100,
            split_gap_frames=5,
            min_duration_frames=6,
            min_points=8,
            min_distance=20.0,
            max_link_speed=60.0,
            max_speed=90.0,
            max_acceleration=55.0,
            max_mean_angle_change=90.0,
            min_score=0.55,
            serve_search_ratio=0.55,
            serve_min_distance=80.0,
            serve_min_speed=4.0,
            toss_min_height=40.0,
            toss_max_x_drift=80.0,
        )

        self.assertEqual([point.frame for point in filtered], list(range(13, 31)) + list(range(60, 81)))
        self.assertEqual({point.tracklet_id for point in filtered}, {2, 4})
        self.assertEqual(report["total_tracklets"], 4)
        self.assertEqual(report["kept_tracklets"], 2)
        self.assertFalse(report["tracklets"][0]["kept"])
        self.assertTrue(report["tracklets"][1]["serve_like"])
        self.assertEqual(len(report["reacquisition_boundaries"]), 3)

    def test_preserves_predicted_occlusion_tracklet_after_serve(self) -> None:
        serve = [ball(frame, 100 + frame * 12, 250 - frame * 2) for frame in range(0, 11)]
        predictions = [
            BallObservation(frame, 232 + (frame - 16) * 3, 228, 5.0, 0.3, "predicted")
            for frame in range(16, 19)
        ]
        continuation = [ball(frame, 250 + (frame - 24) * 5, 228) for frame in range(24, 33)]
        filtered, report = filter_offline_ball_tracklets(
            serve + predictions + continuation,
            frame_count=100,
            split_gap_frames=5,
            min_duration_frames=6,
            min_points=8,
            min_distance=20.0,
            max_link_speed=60.0,
            max_speed=90.0,
            max_acceleration=55.0,
            max_mean_angle_change=90.0,
            min_score=0.90,
            serve_search_ratio=0.55,
            serve_min_distance=80.0,
            serve_min_speed=4.0,
            toss_min_height=40.0,
            toss_max_x_drift=80.0,
        )

        self.assertEqual([point.frame for point in filtered if point.source == "predicted"], [16, 17, 18])
        predicted_analysis = report["tracklets"][1]
        self.assertTrue(predicted_analysis["protected_post_serve_predictions"])
        self.assertEqual(predicted_analysis["decision_reason"], "protected predicted trajectory after serve")


if __name__ == "__main__":
    unittest.main()
