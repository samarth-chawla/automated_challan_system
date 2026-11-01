import unittest
import tempfile
from pathlib import Path
import json

import numpy as np

from speed_estimation import SpeedEstimator
from anpr_core import SessionStore


class SpeedTests(unittest.TestCase):
    def estimator(self, limit=15, fps=25):
        return SpeedEstimator({"mode": "demo", "image_points": [[0, 0], [100, 0], [100, 200], [0, 200]],
                               "road_width_m": 10, "road_length_m": 20, "speed_limit_kmh": limit}, fps)

    def drive(self, estimator, moving=True, jitter=False):
        results = []
        for frame in range(1, 25):
            offset = (-1 if frame % 2 else 1) if jitter else 0
            results.append(estimator.update(1, (50, 20 + frame * 2 + offset), frame, moving))
        return results

    def test_known_distance_and_source_frames_give_18_kmh(self):
        readings = self.drive(self.estimator())
        self.assertAlmostEqual(readings[-1].kmh, 18, delta=0.1)
        self.assertEqual(readings[-1].status, "demo_estimate")

    def test_higher_source_fps_changes_speed_without_using_processing_time(self):
        readings = self.drive(self.estimator(fps=50))
        self.assertAlmostEqual(readings[-1].kmh, 36, delta=0.1)

    def test_overspeed_is_sustained_and_deduplicated(self):
        readings = self.drive(self.estimator())
        self.assertEqual(sum(reading.new_alert for reading in readings), 1)
        self.assertTrue(readings[-1].over_limit)

    def test_under_limit_never_creates_alert(self):
        readings = self.drive(self.estimator(limit=20))
        self.assertFalse(any(reading.new_alert for reading in readings))

    def test_box_movement_without_optical_motion_never_creates_speed_alert(self):
        readings = self.drive(self.estimator(), moving=False)
        self.assertFalse(any(reading.new_alert for reading in readings))
        self.assertTrue(all(reading.kmh is None for reading in readings))

    def test_small_jitter_does_not_create_large_speed(self):
        readings = self.drive(self.estimator(), jitter=True)
        self.assertAlmostEqual(readings[-1].kmh, 18, delta=2)

    def test_tracking_jump_gap_and_geometry_loss_reset_measurement(self):
        estimator = self.estimator()
        self.drive(estimator)
        self.assertIsNone(estimator.update(1, (50, 100), 40, True).kmh)
        self.assertEqual(estimator.update(1, (50, 180), 41, True).status, "tracking_jump")
        self.assertEqual(estimator.update(1, (50, 180), 42, True, reliable=False).status, "alignment_unknown")

    def test_outside_calibrated_region_has_no_fabricated_speed(self):
        self.assertEqual(self.estimator().update(1, (200, 400), 1, True).status, "outside_calibration")

    def test_last_estimate_is_display_only_and_expires(self):
        estimator = self.estimator()
        self.drive(estimator)
        reading = estimator.update(1, (200, 400), 25, True)
        self.assertIsNone(reading.kmh)
        self.assertAlmostEqual(reading.last_kmh, 18, delta=0.1)
        self.assertAlmostEqual(reading.last_age_seconds, 1/25)
        self.assertFalse(reading.new_alert)
        self.assertFalse(reading.over_limit)
        expired = estimator.update(1, (200, 400), 80, True)
        self.assertIsNone(expired.last_kmh)

    def test_last_estimate_cannot_transfer_to_another_vehicle(self):
        estimator = self.estimator()
        self.drive(estimator)
        reading = estimator.update(2, (200, 400), 25, True)
        self.assertIsNone(reading.last_kmh)

    def test_track_jump_discards_last_value(self):
        estimator = self.estimator()
        self.drive(estimator)
        reading = estimator.update(1, (50, 180), 25, True)
        self.assertEqual(reading.status, "tracking_jump")
        self.assertIsNone(reading.last_kmh)

    def test_perspective_plane_not_a_constant_pixel_scale(self):
        import cv2
        calibration = {"mode": "demo", "image_points": [[30, 0], [70, 0], [100, 200], [0, 200]],
                       "road_width_m": 10, "road_length_m": 20, "speed_limit_kmh": 30}
        estimator = SpeedEstimator(calibration, 25)
        inverse = np.linalg.inv(estimator.transform)
        for frame in range(1, 25):
            world = np.float32([[[5, 2 + frame * 0.2]]])
            point = cv2.perspectiveTransform(world, inverse)[0, 0]
            reading = estimator.update(1, point, frame, True)
        self.assertAlmostEqual(reading.kmh, 18, delta=0.1)


class SpeedStoreTests(unittest.TestCase):
    def test_speed_and_red_events_can_coexist_and_share_later_plate_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SessionStore(Path(temporary) / "session", 25, "tr2.mp4")
            record = store.touch(1, "car", [10, 10, 100, 200], 1)
            speeding = store.register_speed_event(1, 10, 60, 50, "demo", "GREEN")
            self.assertEqual(speeding["signal"], "GREEN")
            self.assertEqual(speeding["event_type"], "demo_speed_limit_exceeded")
            self.assertEqual(len(store.events), 0)
            self.assertIs(store.register_speed_event(1, 12, 65, 50, "demo", "GREEN"), speeding)
            red = store.register_event(1, 20, signal="RED")
            record.observe_plate("NN773", 0.95, 30)
            store.flush(30)
            self.assertEqual(speeding["vehicle_number"], "NN773")
            self.assertEqual(red["vehicle_number"], "NN773")
            self.assertEqual(speeding["number_source"], "read_after_event")
            saved = json.loads((store.directory / "speed_violations.json").read_text())
            self.assertEqual(len(saved), 1)
            self.assertTrue((store.directory / "speed_violations.csv").is_file())

    def test_under_limit_cannot_be_saved_as_speeding(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SessionStore(Path(temporary) / "session", 25, "tr2.mp4")
            store.touch(1, "car", [0, 0, 100, 100], 1)
            with self.assertRaises(ValueError):
                store.register_speed_event(1, 10, 30, 50, "demo", "RED")
            self.assertEqual(len(store.speed_events), 0)


if __name__ == "__main__":
    unittest.main()
