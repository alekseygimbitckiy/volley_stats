import argparse
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from classify_rally_serve_receive import (  # noqa: E402
    BallPoint,
    PlayerBox,
    detect_toss_contact_serve,
    find_serve_contact_player,
    refine_serve_contact_and_type,
)


def ball(frame: int, x: float, y: float) -> BallPoint:
    return BallPoint(frame, x, y, 5.0, 1.0)


def args() -> argparse.Namespace:
    return argparse.Namespace(
        serve_toss_search=45,
        serve_contact_max_gap=10,
        serve_contact_flight=30,
        serve_contact_min_speed=4.0,
        serve_toss_max_x_drift=80.0,
        serve_toss_min_vertical_motion=6.0,
        serve_toss_min_vertical_x_ratio=2.0,
        serve_power_toss_min_height=70.0,
        serve_power_min_speed=20.0,
        serve_window=14,
        serve_min_distance=120.0,
        serve_min_speed=8.0,
        serve_max_mean_angle_change=38.0,
        serve_search_ratio=0.8,
        reception_window=4,
    )


def initial_serve() -> dict:
    return {
        "frame": 4,
        "x": 100.0,
        "y": 180.0,
        "window_end_frame": 18,
        "window_end_x": 520.0,
        "window_end_y": 110.0,
        "direction": "toward_far_team",
        "y_speed_px_per_frame": -5.0,
        "avg_speed_px_per_frame": 30.0,
        "total_distance_px": 420.0,
        "mean_angle_change_deg": 0.0,
    }


def flight(start_frame: int = 6) -> list[BallPoint]:
    return [ball(frame, 160 + (frame - start_frame) * 30, 170 - (frame - start_frame) * 5) for frame in range(start_frame, 21)]


class ServeContactDetectionTests(unittest.TestCase):
    def test_high_vertical_toss_is_power_and_contact_excludes_toss(self) -> None:
        points = [
            ball(0, 100, 200),
            ball(1, 102, 160),
            ball(2, 99, 120),
            ball(3, 101, 145),
            ball(4, 100, 180),
            *flight(),
        ]
        result = refine_serve_contact_and_type(points, initial_serve(), args())
        self.assertTrue(result["contact_captured"])
        self.assertEqual(result["frame"], 4)
        self.assertEqual(result["flight_start_frame"], 6)
        self.assertEqual(result["serve_type"], "power")
        self.assertTrue(result["toss"]["complete_arc"])

    def test_low_compact_toss_is_floater(self) -> None:
        points = [
            ball(0, 100, 200),
            ball(1, 102, 190),
            ball(2, 99, 180),
            ball(3, 101, 190),
            ball(4, 100, 200),
            *flight(),
        ]
        result = refine_serve_contact_and_type(points, initial_serve(), args())
        self.assertTrue(result["contact_captured"])
        self.assertEqual(result["serve_type"], "floater")
        self.assertLess(result["toss"]["vertical_range_px"], 70.0)

    def test_missing_toss_keeps_flight_fallback(self) -> None:
        serve = initial_serve()
        points = flight(start_frame=4)
        result = refine_serve_contact_and_type(points, serve, args())
        self.assertFalse(result["contact_captured"])
        self.assertEqual(result["serve_type"], "unknown")
        self.assertEqual(result["frame"], 4)

    def test_contact_first_detector_does_not_consume_serve_as_reception(self) -> None:
        points = [
            ball(0, 100, 200),
            ball(1, 101, 160),
            ball(2, 99, 120),
            ball(3, 100, 150),
            ball(4, 100, 190),
            *[ball(frame, 100 + (frame - 4) * 9, 190 + (frame - 4) * 5) for frame in range(5, 21)],
        ]
        candidates = [
            {"frame": 4, "angle_change_deg": 170.0, "speed_before_px_per_frame": 8.0, "speed_after_px_per_frame": 10.0},
            {"frame": 21, "angle_change_deg": 150.0, "speed_before_px_per_frame": 10.0, "speed_after_px_per_frame": 7.0},
        ]
        result = detect_toss_contact_serve(points, candidates, frame_count=60, args=args())
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["frame"], 4)
        self.assertEqual(result["window_end_frame"], 20)
        self.assertEqual(result["serve_type"], "power")
        self.assertTrue(result["contact_captured"])

    def test_server_assignment_uses_opponent_even_when_near_player_is_closer(self) -> None:
        near = PlayerBox(10, "near-1", "jersey_12", (90, 100, 130, 220), 0.9, team="near")
        opponent = PlayerBox(10, "far-1", None, (80, 115, 120, 230), 0.8, team="opponent")
        result = find_serve_contact_player(
            cv2=None,
            video_path=Path("unused.mp4"),
            player_boxes_by_frame={10: [near, opponent]},
            ball_x=100,
            ball_y=90,
            contact_frame=10,
            serving_team="far",
            frame_window=0,
            max_distance=220,
            pose_classifier=None,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["track_id"], "far-1")
        self.assertEqual(result["team"], "opponent")


if __name__ == "__main__":
    unittest.main()
