import json
from pathlib import Path
import unittest

from detection_overlay import overlap, place_labels, speed_caption
from speed_estimation import SpeedReading


class OverlayTests(unittest.TestCase):
    def test_current_stale_and_missing_speeds_have_explicit_captions(self):
        self.assertIn("9.0 km/h", speed_caption(SpeedReading(kmh=9, status="demo_estimate"))[0])
        value = SpeedReading(status="outside_calibration", last_kmh=13.5, last_age_seconds=1.4)
        self.assertIn("Last: 13.5", speed_caption(value)[0])
        self.assertIn("1.4s ago", speed_caption(value)[0])
        self.assertIn("motion uncertain", speed_caption(SpeedReading(status="motion_unconfirmed"))[0])
        self.assertIn("outside road zone", speed_caption(SpeedReading(status="outside_calibration"))[0])

    def test_frame215_labels_are_in_bounds_and_do_not_overlap(self):
        vehicles = json.loads((Path(__file__).parent / "tests/fixtures/frame215_tracks.json").read_text())
        items = [{"box": vehicle["box"],
                  "lines": [f"#{vehicle['id']} PLATE READING | CHECKING", "Speed: collecting frames"],
                  "color": (50, 220, 130)} for vehicle in vehicles]
        placed = place_labels(items)
        self.assertEqual(len(placed), len(vehicles))
        rectangles = [rectangle for _, rectangle in placed]
        for index, rectangle in enumerate(rectangles):
            self.assertGreaterEqual(rectangle[0], 0)
            self.assertGreaterEqual(rectangle[1], 84)
            self.assertLessEqual(rectangle[2], 1100)
            self.assertLessEqual(rectangle[3], 700)
            self.assertEqual(overlap(rectangle, (985, 0, 1100, 230)), 0)
            for previous in rectangles[:index]:
                self.assertEqual(overlap(rectangle, previous), 0)


if __name__ == "__main__":
    unittest.main()
