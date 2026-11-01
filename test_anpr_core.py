import csv
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from anpr_core import (CrossingMonitor, GreenLight, RedEventMonitor, RedLight, SessionStore,
                       VehicleRecord, event_display_state, normalize_plate, plate_owner, signal_state)
from road_boundary import RoadBoundary
from vehicle_motion import VehicleMotion


class CrossingTests(unittest.TestCase):
    def monitor(self):
        return CrossingMonitor(line=[(220, 420), (880, 420)])

    def test_crossing_during_red_only_once(self):
        monitor = self.monitor()
        self.assertFalse(monitor.observe(1, (500, 400), 1, "RED", moving=True))
        self.assertTrue(monitor.observe(1, (505, 440), 2, "RED", moving=True))
        self.assertFalse(monitor.observe(1, (505, 450), 3, "RED", moving=True))
        monitor.observe(1, (500, 400), 4, "RED", moving=True)
        self.assertFalse(monitor.observe(1, (500, 440), 5, "RED", moving=True))

    def test_green_crossing_does_not_become_red_violation_later(self):
        monitor = self.monitor()
        monitor.observe(1, (500, 400), 1, "GREEN", moving=True)
        self.assertFalse(monitor.observe(1, (500, 440), 2, "GREEN", moving=True))
        self.assertFalse(monitor.observe(1, (500, 460), 3, "RED", moving=True))

    def test_first_seen_beyond_line_and_long_gap_are_not_crossings(self):
        monitor = self.monitor()
        self.assertFalse(monitor.observe(1, (500, 450), 1, "RED", moving=True))
        monitor.observe(2, (500, 400), 1, "RED", moving=True)
        self.assertFalse(monitor.observe(2, (500, 450), 40, "RED", moving=True))

    def test_direction_span_and_unknown_signal(self):
        for point, signal in [((100, 440), "RED"), ((500, 440), "UNKNOWN")]:
            monitor = self.monitor()
            monitor.observe(1, (point[0], 400), 1, "GREEN", moving=True)
            self.assertFalse(monitor.observe(1, point, 2, signal, moving=True))
        monitor = self.monitor()
        monitor.observe(1, (500, 450), 1, "RED", moving=True)
        self.assertFalse(monitor.observe(1, (500, 400), 2, "RED", moving=True))

    def test_hysteresis_preserves_approach_side(self):
        monitor = self.monitor()
        monitor.observe(1, (500, 400), 1, "RED", moving=True)
        for frame, y in enumerate([418, 421, 419, 423], 2):
            self.assertFalse(monitor.observe(1, (500, y), frame, "RED", moving=True))
        self.assertTrue(monitor.observe(1, (500, 440), 6, "RED", moving=True))

    def test_stationary_bus_box_expansion_is_not_crossing(self):
        monitor = self.monitor()
        monitor.observe(80, (500, 400), 1, "RED", moving=False)
        self.assertFalse(monitor.observe(80, (500, 470), 2, "RED", moving=False))
        self.assertNotIn(80, monitor.flagged)

    def test_sloping_road_line_uses_segment_geometry(self):
        monitor = CrossingMonitor(line=[(200, 600), (1000, 500)])
        self.assertFalse(monitor.observe(1, (600, 535), 1, "RED", moving=True))
        self.assertTrue(monitor.observe(1, (600, 565), 2, "RED", moving=True))


class DisplayAndRoadTests(unittest.TestCase):
    def test_green_and_unknown_never_display_a_past_event_as_active_red(self):
        event = {"frame": 20}
        self.assertEqual(event_display_state(event, "GREEN", 1), "PAST")
        self.assertEqual(event_display_state(event, "UNKNOWN", 1), "PAST")
        self.assertEqual(event_display_state(event, "RED", 25), "PAST")
        self.assertEqual(event_display_state(event, "RED", 10), "ACTIVE")
        self.assertIsNone(event_display_state(None, "RED", 1))

    def board(self, center):
        image = np.zeros((700, 1100, 3), dtype=np.uint8)
        cv2.ellipse(image, center, (35, 42), 0, 0, 360, (0, 0, 255), 6)
        return image

    def test_line_moves_with_board_and_point_coordinates_remain_stable(self):
        boundary = RoadBoundary([(190, 635), (1080, 620)])
        first = boundary.update(self.board((187, 185)))
        point1 = boundary.to_reference((600, 610))
        second = boundary.update(self.board((207, 145)))
        point2 = boundary.to_reference((620, 570))
        np.testing.assert_allclose(np.array(second)-first, [[20, -40], [20, -40]], atol=1)
        np.testing.assert_allclose(point1, point2, atol=1)
        self.assertTrue(boundary.reliable)

    def test_missing_board_disables_geometry_after_short_gap(self):
        boundary = RoadBoundary([(190, 635), (1080, 620)])
        boundary.update(self.board((187, 185)))
        for _ in range(11):
            boundary.update(np.zeros((700, 1100, 3), np.uint8))
        self.assertFalse(boundary.reliable)

    def test_inactive_red_lamp_outline_is_neutral_on_green(self):
        from types import SimpleNamespace
        from anpr_demo import render
        image = np.zeros((700, 1100, 3), np.uint8)
        store = SimpleNamespace(vehicles={}, events={})
        result = render(image, store, {}, {}, "GREEN", 98, 25,
                        [(190, 635), (1080, 620)], 1.0, {})
        blue, green, red = map(int, result[55, 1055])
        self.assertEqual(blue, green)
        self.assertEqual(green, red)

    def test_past_vehicle_box_is_amber_on_green_and_red_only_in_current_red_phase(self):
        from types import SimpleNamespace
        from anpr_demo import render
        from vehicle_motion import Motion
        record = VehicleRecord(1, "car", 1, 98, box=[100, 200, 250, 400])
        event = {"vehicle_id": 1, "frame": 20, "video_seconds": 0.8,
                 "vehicle_number": None, "event_type": "demo_red_zone_movement"}
        store = SimpleNamespace(vehicles={1: record}, events={1: event})
        arguments = (np.zeros((700, 1100, 3), np.uint8), store,
                     {1: record.box}, {}, "GREEN", 98, 25,
                     [(190, 635), (1080, 620)], 1.0, {1: Motion()})
        result = render(*arguments, red_phase_start=90)
        np.testing.assert_array_equal(result[250, 100], [40, 190, 240])
        changed = list(arguments)
        changed[4] = "RED"
        result = render(*changed, red_phase_start=10)
        np.testing.assert_array_equal(result[250, 100], [30, 50, 245])


class RedZoneTests(unittest.TestCase):
    def test_moving_car_already_beyond_boundary_is_zone_alert_not_claimed_crossing(self):
        monitor = RedEventMonitor(line=[(220, 420), (880, 420)])
        for frame in range(1, 3):
            self.assertIsNone(monitor.observe(1, (500, 550), frame, "RED", moving=True))
        self.assertEqual(monitor.observe(1, (500, 560), 3, "RED", moving=True), "demo_red_zone_movement")
        self.assertIsNone(monitor.observe(1, (500, 570), 4, "RED", moving=True))

    def test_stopped_bus_and_green_traffic_are_not_alerts(self):
        monitor = RedEventMonitor(line=[(220, 420), (880, 420)])
        for frame in range(1, 15):
            self.assertIsNone(monitor.observe(1, (500, 550 + frame), frame, "RED", moving=False))
            self.assertIsNone(monitor.observe(2, (600, 550 + frame), frame, "GREEN", moving=True))

    def test_observed_crossing_keeps_its_distinct_event_type(self):
        monitor = RedEventMonitor(line=[(220, 420), (880, 420)])
        monitor.observe(1, (500, 400), 1, "RED", moving=True)
        self.assertEqual(monitor.observe(1, (500, 450), 2, "RED", moving=True), "demo_red_line_crossing")

    def test_strict_mode_does_not_count_already_beyond_line(self):
        monitor = RedEventMonitor(line=[(220, 420), (880, 420)], mode="crossing")
        for frame in range(1, 15):
            self.assertIsNone(monitor.observe(1, (500, 550+frame), frame, "RED", moving=True))

    def test_each_vehicle_has_its_own_counter_and_gaps_reset_evidence(self):
        monitor = RedEventMonitor(line=[(220, 420), (880, 420)])
        for frame in range(1, 3):
            monitor.observe(1, (500, 550), frame, "RED", moving=True)
        self.assertIsNone(monitor.observe(1, (500, 550), 20, "RED", moving=True))
        for frame in range(21, 23):
            result = monitor.observe(1, (500, 550), frame, "RED", moving=True)
        self.assertEqual(result, "demo_red_zone_movement")
        for frame in range(1, 4):
            result = monitor.observe(2, (600, 550), frame, "RED", moving=True)
        self.assertEqual(result, "demo_red_zone_movement")

    def test_yellow_audi_red_motion_before_green_is_not_missed(self):
        data = json.loads((Path(__file__).parent / "tests/fixtures/yellow_audi_red_phase.json").read_text())
        monitor = RedEventMonitor(line=data["line"])
        alerts = []
        for row in data["observations"]:
            kind = monitor.observe(230, row["point"], row["frame"], row["signal"], moving=row["moving"])
            if kind:
                alerts.append((row["frame"], row["signal"], kind))
        self.assertEqual(alerts, [(data["expected_alert_frame"], "RED", "demo_red_zone_movement")])


class MotionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.background = rng.integers(20, 65, (250, 300, 3), dtype=np.uint8)
        self.texture = rng.integers(40, 230, (80, 100, 3), dtype=np.uint8)

    def frame(self, y=80):
        image = self.background.copy()
        image[y:y+80, 80:180] = self.texture
        return image

    def test_static_image_with_expanding_box_stays_stopped(self):
        estimator = VehicleMotion()
        for frame in range(12):
            result = estimator.update(self.frame(), {1: [75, 75, 185, 165 + frame * 3]}, frame)
        self.assertFalse(result[1].moving)
        self.assertGreaterEqual(result[1].samples, 4)
        self.assertLess(abs(result[1].dy), 1)

    def test_car_moving_down_has_independent_motion_evidence(self):
        estimator = VehicleMotion()
        for frame in range(10):
            y = 80 + frame * 3
            result = estimator.update(self.frame(y), {1: [75, y-5, 185, y+85]}, frame)
        self.assertTrue(result[1].moving)
        self.assertGreater(result[1].dy, 15)

    def test_untextured_or_newly_appearing_vehicle_is_unknown(self):
        estimator = VehicleMotion()
        image = np.zeros((250, 300, 3), dtype=np.uint8)
        estimator.update(image, {1: [50, 50, 150, 150]}, 1)
        result = estimator.update(image, {1: [50, 50, 150, 200]}, 2)
        self.assertFalse(result[1].moving)
        self.assertEqual(result[1].label, "CHECKING")

    def test_diagonally_approaching_car_is_not_required_to_move_straight_down(self):
        estimator = VehicleMotion()
        for frame in range(10):
            transform = np.float32([[1, 0, -frame * 2], [0, 1, frame * 0.3]])
            image = self.background.copy()
            foreground = np.zeros_like(image)
            foreground[80:160, 80:180] = self.texture
            shifted = cv2.warpAffine(foreground, transform, (300, 250))
            mask = cv2.warpAffine(np.pad(np.full((80, 100), 255, np.uint8),
                                       ((80, 90), (80, 120))), transform, (300, 250))
            image[mask > 0] = shifted[mask > 0]
            result = estimator.update(image, {1: [75-frame*2, 75+frame*0.3,
                                                  185-frame*2, 165+frame*0.3]}, frame)
        self.assertTrue(result[1].moving)

    def test_passing_car_features_do_not_move_stationary_bus(self):
        estimator = VehicleMotion()
        car = np.random.default_rng(3).integers(40, 230, (70, 70, 3), dtype=np.uint8)
        for frame in range(10):
            image = self.frame()
            x, y = 45 + frame * 2, 125 + frame * 2
            image[y:y+70, x:x+70] = car
            result = estimator.update(image, {1: [75, 75, 185, 165],
                                               2: [x, y, x+70, y+70]}, frame)
        self.assertFalse(result[1].moving)
        self.assertGreaterEqual(result[1].samples, 4)

    def test_camera_translation_does_not_move_parked_vehicle(self):
        estimator = VehicleMotion()
        for frame in range(10):
            transform = np.float32([[1, 0, frame], [0, 1, frame]])
            image = cv2.warpAffine(self.frame(), transform, (300, 250))
            result = estimator.update(image, {1: [75+frame, 75+frame, 185+frame, 165+frame]}, frame)
        self.assertFalse(result[1].moving)
        self.assertLess(abs(result[1].dy), 1)


class PlateMemoryTests(unittest.TestCase):
    def test_filter_and_repeated_consensus(self):
        self.assertEqual(normalize_plate("kw 527"), "KW527")
        self.assertIsNone(normalize_plate("VIDEVO"))
        self.assertIsNone(normalize_plate("123456"))
        self.assertIsNone(normalize_plate("S5"))
        record = VehicleRecord(1, "car", 1, 1)
        self.assertFalse(record.observe_plate("KW527", 0.2, 1))
        record.observe_plate("KH527", 0.5, 1)
        record.observe_plate("KW527", 0.8, 2)
        record.observe_plate("KW527", 0.8, 3)
        record.observe_plate("KW527", 0.8, 3)
        self.assertEqual(record.number, "KW527")
        self.assertEqual(record.plate["observations"], 2)
        self.assertEqual(record.plate["plate_status"], "repeated_read")

    def test_plate_belongs_to_one_vehicle(self):
        vehicles = {1: [0, 0, 200, 200], 2: [100, 40, 180, 180]}
        self.assertEqual(plate_owner([120, 150, 150, 170], vehicles), 2)
        self.assertIsNone(plate_owner([250, 250, 280, 270], vehicles))
        self.assertIsNone(plate_owner([110, 45, 140, 55], {2: vehicles[2]}))

    def test_cached_plate_is_reused_and_late_plate_has_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SessionStore(Path(temporary) / "session", 25, "tr2.mp4")
            with self.assertRaises(ValueError):
                store.register_event(99, 1, signal="GREEN")
            self.assertEqual(len(store.events), 0)
            first = store.touch(1, "car", [0, 0, 10, 10], 1)
            first.observe_plate("KW527", 0.9, 2)
            event = store.register_event(1, 20)
            self.assertEqual(event["vehicle_number"], "KW527")
            self.assertEqual(event["number_source"], "cached_before_crossing")
            self.assertIs(store.register_event(1, 21), event)
            second = store.touch(2, "car", [0, 0, 10, 10], 1)
            late_event = store.register_event(2, 20)
            self.assertIsNone(late_event["vehicle_number"])
            second.observe_plate("YB6433", 0.9, 30)
            store.flush(30)
            self.assertEqual(late_event["vehicle_number"], "YB6433")
            self.assertEqual(late_event["number_source"], "read_after_crossing")
            self.assertIsNone(late_event["number_at_crossing"])
            self.assertEqual(event["vehicle_number"], "KW527")
            saved = json.loads((store.directory / "violations.json").read_text())
            self.assertEqual(len(saved), 2)
            with (store.directory / "vehicles.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)


class ViolationExportTests(unittest.TestCase):
    def test_all_categories_are_persisted_in_json_and_csv_with_late_plate_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SessionStore(Path(temporary) / "session", 25, "tr2.mp4")
            car = store.touch(1, "car", [0, 0, 10, 10], 1)
            store.touch(2, "bus", [20, 0, 30, 10], 1)
            store.touch(3, "truck", [40, 0, 50, 10], 1)
            store.register_event(1, 30)
            store.register_event(2, 20, event_type="demo_red_zone_movement")
            store.register_speed_event(1, 10, 60, 50, "demo", "GREEN")
            store.register_speed_event(3, 40, 60, 50, "calibrated", "RED")
            store.flush(40)
            car.observe_plate("KW527", 0.9, 45)
            store.flush(45)

            saved = json.loads((store.directory / "all_violations.json").read_text())
            self.assertEqual([event["frame"] for event in saved], [10, 20, 30, 40])
            self.assertEqual([event["violation_type"] for event in saved],
                             ["speeding", "red_zone_movement", "red_light_crossing", "speeding"])
            self.assertEqual(saved[0]["event_type"], "demo_speed_limit_exceeded")
            self.assertEqual(saved[3]["event_type"], "estimated_speed_limit_exceeded")
            for event in (saved[0], saved[2]):
                self.assertEqual(event["vehicle_number"], "KW527")
                self.assertIsNone(event["number_at_event"])
                self.assertTrue(event["number_source"].startswith("read_after_"))
                self.assertTrue(event["evidence_image"].endswith(".jpg"))

            for stem, expected in [("all_violations", saved),
                                   ("violations", [saved[2], saved[1]]),
                                   ("speed_violations", [saved[0], saved[3]])]:
                data = json.loads((store.directory / f"{stem}.json").read_text())
                self.assertEqual(data, expected)
                with (store.directory / f"{stem}.csv").open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), len(expected))
                for row, event in zip(rows, expected):
                    self.assertEqual(row["violation_type"], event["violation_type"])
                    self.assertEqual(row["event_type"], event["event_type"])
                    self.assertEqual(row["vehicle_number"], event["vehicle_number"] or "")
                    self.assertEqual(row["evidence_image"], event["evidence_image"])

    def test_no_alerts_export_an_empty_list_and_csv_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SessionStore(Path(temporary) / "session", 25, "tr2.mp4")
            store.flush(0)
            for stem in ("violations", "speed_violations", "all_violations"):
                self.assertEqual(json.loads((store.directory / f"{stem}.json").read_text()), [])
                with (store.directory / f"{stem}.csv").open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    self.assertIn("violation_type", reader.fieldnames)
                    self.assertEqual(list(reader), [])

    def test_unknown_event_cannot_be_saved_with_a_misleading_category(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SessionStore(Path(temporary) / "session", 25, "tr2.mp4")
            store.touch(1, "car", [0, 0, 10, 10], 1)
            for kind in ("typo", "demo_speed_limit_exceeded"):
                with self.assertRaises(ValueError):
                    store.register_event(1, 20, event_type=kind)
            self.assertEqual(store.events, {})


class SignalTests(unittest.TestCase):
    def test_actual_frame215_is_yellow_not_unknown_or_red(self):
        image = np.zeros((700, 1100, 3), np.uint8)
        crop = cv2.imread(str(Path(__file__).parent / "tests/fixtures/yellow_signal_frame215.png"))
        self.assertIsNotNone(crop)
        image[10:210, 1000:1080] = crop
        self.assertEqual(signal_state(image), "YELLOW")

    def test_actual_frame98_lamp_is_green(self):
        image = np.zeros((700, 1100, 3), np.uint8)
        crop = cv2.imread(str(Path(__file__).parent / "tests/fixtures/green_signal_frame98.png"))
        self.assertIsNotNone(crop)
        image[10:210, 1000:1080] = crop
        self.assertEqual(signal_state(image), "GREEN")

    def test_saturated_lamps_and_white_disabled_lights(self):
        frame = np.zeros((700, 1100, 3), dtype=np.uint8)
        cv2.fillPoly(frame, [RedLight], (180, 180, 180))
        self.assertEqual(signal_state(frame), "UNKNOWN")
        cv2.fillPoly(frame, [GreenLight], (0, 255, 0))
        self.assertEqual(signal_state(frame), "GREEN")
        cv2.fillPoly(frame, [GreenLight], (180, 180, 180))
        cv2.fillPoly(frame, [RedLight], (0, 0, 255))
        self.assertEqual(signal_state(frame), "RED")


if __name__ == "__main__":
    unittest.main()
