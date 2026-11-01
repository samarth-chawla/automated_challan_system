"""Vehicle/plate memory and crossing rules, independent of neural-network inference."""

from dataclasses import dataclass, field
from collections import deque
import csv
import json
from pathlib import Path
import re

import cv2
import numpy as np


# Lamp ellipses calibrated against the native tr2.mp4 frames, at 1100 x 700.
FRAME_SIZE = (1100, 700)
RedLight = cv2.ellipse2Poly((1038, 55), (17, 22), 0, 0, 360, 15)
YellowLight = cv2.ellipse2Poly((1038, 111), (17, 22), 0, 0, 360, 15)
GreenLight = cv2.ellipse2Poly((1038, 166), (17, 22), 0, 0, 360, 15)
ORIGINAL_ROI_line = [(220, 420), (880, 420)]
# Approximate road-level transverse line at the sign pole's base in reference frame 98.
# RoadBoundary tracks the sign to translate this line with the camera view.
ROI_line = [(190, 635), (1080, 620)]
# tr2.mp4 shows Hong Kong plates. This filter is configurable for other footage.
PLATE_PATTERN = r"(?=.{4,6}$)[A-Z]{1,2}[0-9]{1,4}"

# Stable export categories; event_type retains the detailed demo/estimate meaning.
RED_VIOLATION_TYPES = {
    "demo_red_line_crossing": "red_light_crossing",
    "demo_red_zone_movement": "red_zone_movement",
}


def normalize_plate(text, pattern=PLATE_PATTERN):
    value = re.sub(r"[^A-Z0-9]", "", text.upper())
    return value if re.fullmatch(pattern, value) else None


def event_display_state(event, signal, red_phase_start=0):
    if event is None:
        return None
    return "ACTIVE" if signal == "RED" and event["frame"] >= red_phase_start else "PAST"


def signal_state(frame):
    """Read saturated lamp colors from the unannotated frame, not brightness."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def fraction(polygon, ranges):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        color = np.zeros_like(mask)
        for lower, upper in ranges:
            color |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        return np.count_nonzero(color & mask) / max(1, np.count_nonzero(mask))

    red = fraction(RedLight, [((0, 100, 100), (10, 255, 255)),
                              ((170, 100, 100), (179, 255, 255))])
    green = fraction(GreenLight, [((35, 100, 100), (90, 255, 255))])
    yellow = fraction(YellowLight, [((15, 100, 100), (35, 255, 255))])
    if yellow >= 0.08 and yellow > max(red, green) * 1.3:
        return "YELLOW"
    if red >= 0.08 and red > green * 1.3:
        return "RED"
    if green >= 0.08 and green > red * 1.3:
        return "GREEN"
    return "UNKNOWN"


def plate_owner(plate_box, vehicles):
    """Assign a plate to one containing vehicle; prefer the smallest container."""
    px1, py1, px2, py2 = plate_box
    area = max(1, (px2 - px1) * (py2 - py1))
    cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
    owners = []
    for vehicle_id, box in vehicles.items():
        x1, y1, x2, y2 = box
        intersection = max(0, min(px2, x2) - max(px1, x1)) * max(0, min(py2, y2) - max(py1, y1))
        if x1 <= cx <= x2 and y1 + (y2 - y1) * 0.35 <= cy <= y2 and intersection / area >= 0.8:
            owners.append(((x2 - x1) * (y2 - y1), vehicle_id))
    return min(owners)[1] if owners else None


@dataclass
class PlateVote:
    count: int = 0
    weight: float = 0.0
    confidence: float = 0.0
    first_frame: int = 0
    last_frame: int = 0
    image: str = ""


@dataclass
class VehicleRecord:
    vehicle_id: int
    vehicle_type: str
    first_frame: int
    last_frame: int
    box: list = field(default_factory=list)
    candidates: dict = field(default_factory=dict)
    last_ocr_frame: int = -1000
    speed_kmh: float | None = None
    peak_speed_kmh: float | None = None
    speed_status: str = "collecting"
    speed_limit_kmh: float | None = None
    speed_calibration: str = "demo"
    last_speed_kmh: float | None = None
    last_speed_age_seconds: float | None = None

    def observe_plate(self, text, confidence, frame, image="", pattern=PLATE_PATTERN, quality=1.0):
        value = normalize_plate(text, pattern)
        if not value or confidence < 0.35:
            return False
        vote = self.candidates.setdefault(value, PlateVote(first_frame=frame))
        # Each frame contributes once, even if two OCR preprocessing passes agree.
        if vote.last_frame == frame:
            return False
        vote.count += 1
        vote.weight += confidence ** 2 * quality
        vote.last_frame = frame
        if confidence >= vote.confidence:
            vote.confidence = float(confidence)
            vote.image = image
        return True

    @property
    def number(self):
        return max(self.candidates, key=lambda key: self.candidates[key].weight, default=None)

    @property
    def plate(self):
        number = self.number
        if number is None:
            return {"vehicle_number": None, "plate_status": "unread", "ocr_confidence": 0.0,
                    "observations": 0, "first_read_frame": None, "plate_image": ""}
        vote = self.candidates[number]
        return {"vehicle_number": number,
                "plate_status": "repeated_read" if vote.count >= 2 else "single_read",
                "ocr_confidence": round(vote.confidence, 4), "observations": vote.count,
                "first_read_frame": vote.first_frame, "plate_image": vote.image}

    def snapshot(self):
        return {"vehicle_id": self.vehicle_id, "vehicle_type": self.vehicle_type,
                "first_frame": self.first_frame, "last_frame": self.last_frame,
                "speed_kmh": self.speed_kmh, "peak_speed_kmh": self.peak_speed_kmh,
                "speed_status": self.speed_status, "speed_limit_kmh": self.speed_limit_kmh,
                "speed_calibration": self.speed_calibration,
                "last_speed_kmh": self.last_speed_kmh,
                "last_speed_age_seconds": self.last_speed_age_seconds,
                **self.plate}


class CrossingMonitor:
    """One event per ID, only on a directional crossing while the signal is red."""

    def __init__(self, line=ROI_line, direction="down", margin=4, max_gap=30):
        self.line = line
        self.direction = direction
        self.margin = margin
        self.max_gap = max_gap
        self.positions = {}
        self.flagged = set()

    def observe(self, vehicle_id, point, frame, signal, moving=False):
        x, y = point
        first, second = sorted(self.line, key=lambda value: value[0])
        ax, ay = first
        bx, by = second
        length = np.hypot(bx - ax, by - ay)
        if length == 0:
            return False
        distance = ((bx - ax) * (y - ay) - (by - ay) * (x - ax)) / length
        side = -1 if distance < -self.margin else (1 if distance > self.margin else 0)
        previous = self.positions.get(vehicle_id)
        if previous and frame - previous[2] > self.max_gap:
            previous = None
            self.positions.pop(vehicle_id, None)
        if side == 0:
            return False
        self.positions[vehicle_id] = (side, (x, y), frame, distance)
        if previous is None or previous[0] == side or vehicle_id in self.flagged:
            return False
        target_side = 1 if self.direction == "down" else -1
        if side != target_side or signal != "RED" or not moving:
            return False
        px, py = previous[1]
        fraction = -previous[3] / (distance - previous[3])
        intersection = np.array([px + (x - px) * fraction, py + (y - py) * fraction])
        along = np.dot(intersection - first, np.array(second) - first) / length ** 2
        if not 0 <= along <= 1:
            return False
        self.flagged.add(vehicle_id)
        return True


class RedEventMonitor:
    """Distinguish a captured crossing from movement first seen past the boundary."""

    def __init__(self, line=ROI_line, direction="down", mode="red-zone"):
        self.crossings = CrossingMonitor(line=line, direction=direction)
        self.mode = mode
        self.flagged = set()
        self.history = {}
        left, right = sorted(line, key=lambda point: point[0])
        self.zone = np.array([left, right, (1200, 800), (-100, 800)], dtype=np.int32)

    def observe(self, vehicle_id, point, frame, signal, moving=False):
        crossed = self.crossings.observe(vehicle_id, point, frame, signal, moving=moving)
        if vehicle_id in self.flagged:
            return None
        if crossed:
            self.flagged.add(vehicle_id)
            return "demo_red_line_crossing"
        if self.mode == "crossing":
            return None
        in_zone = cv2.pointPolygonTest(self.zone, (float(point[0]), float(point[1])), False) >= 0
        history = self.history.setdefault(vehicle_id, deque(maxlen=10))
        if signal != "RED" or not in_zone:
            history.clear()
            return None
        if history and frame - history[-1][0] > 2:
            history.clear()
        history.append((frame, moving))
        # Motion already needs four independent optical-flow samples. Three further
        # positive frames retain confirmation without missing the end of a red phase.
        if len(history) >= 3 and sum(value[1] for value in list(history)[-3:]) == 3:
            self.flagged.add(vehicle_id)
            return "demo_red_zone_movement"
        return None


class SessionStore:
    """Per-run cache; a late OCR read can enrich an event for the same tracker ID."""

    def __init__(self, directory, fps, source):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "plates").mkdir()
        (self.directory / "violations").mkdir()
        (self.directory / "speed_violations").mkdir()
        self.fps = fps
        self.source = str(source)
        self.vehicles = {}
        self.events = {}
        self.speed_events = {}

    def touch(self, vehicle_id, vehicle_type, box, frame):
        record = self.vehicles.setdefault(vehicle_id, VehicleRecord(vehicle_id, vehicle_type, frame, frame))
        record.last_frame = frame
        record.box = list(box)
        return record

    def register_event(self, vehicle_id, frame, event_type="demo_red_line_crossing", signal="RED"):
        if signal != "RED":
            raise ValueError("A red event cannot be registered unless the measured signal is RED")
        if event_type not in RED_VIOLATION_TYPES:
            raise ValueError(f"Unsupported red event type: {event_type}")
        if vehicle_id in self.events:
            return self.events[vehicle_id]
        record = self.vehicles[vehicle_id]
        event = {"vehicle_id": vehicle_id, "vehicle_type": record.vehicle_type,
                 "frame": frame, "video_seconds": round((frame - 1) / self.fps, 3),
                 "signal": "RED", "event_type": event_type,
                 "violation_type": RED_VIOLATION_TYPES[event_type],
                 "number_at_event": record.number,
                 "number_at_crossing": record.number if event_type == "demo_red_line_crossing" else None,
                 **record.plate,
                 "number_source": "unread", "evidence_image": f"violations/vehicle_{vehicle_id}_frame_{frame}.jpg"}
        self.events[vehicle_id] = event
        self.refresh_event(record)
        return event

    def refresh_event(self, record):
        for table in [self.events, self.speed_events]:
            if record.vehicle_id not in table:
                continue
            event = table[record.vehicle_id]
            event.update(record.plate)
            first_read = record.plate["first_read_frame"]
            boundary = "crossing" if event["event_type"] == "demo_red_line_crossing" else "event"
            event["number_source"] = ("unread" if first_read is None else
                                      f"cached_before_{boundary}" if first_read < event["frame"] else
                                      f"read_at_{boundary}" if first_read == event["frame"] else f"read_after_{boundary}")

    def register_speed_event(self, vehicle_id, frame, speed, limit, calibration, signal):
        if vehicle_id in self.speed_events:
            return self.speed_events[vehicle_id]
        if not np.isfinite(speed) or speed <= limit:
            raise ValueError("A speeding alert requires a speed above the configured limit")
        record = self.vehicles[vehicle_id]
        event = {"vehicle_id": vehicle_id, "vehicle_type": record.vehicle_type,
                 "frame": frame, "video_seconds": round((frame - 1) / self.fps, 3),
                 "event_type": "demo_speed_limit_exceeded" if calibration == "demo" else "estimated_speed_limit_exceeded",
                 "violation_type": "speeding",
                 "signal": signal, "estimated_speed_kmh": round(speed, 2), "speed_limit_kmh": limit,
                 "calibration_status": calibration, "number_at_event": record.number, **record.plate,
                 "evidence_image": f"speed_violations/vehicle_{vehicle_id}_frame_{frame}.jpg"}
        self.speed_events[vehicle_id] = event
        self.refresh_event(record)
        return event

    def flush(self, frame):
        for record in self.vehicles.values():
            self.refresh_event(record)
        vehicles = [record.snapshot() for record in self.vehicles.values()]
        events = list(self.events.values())
        speed_events = list(self.speed_events.values())
        all_events = sorted(events + speed_events, key=lambda event: (event["frame"], event["vehicle_id"], event["event_type"]))
        payloads = {"vehicle_cache.json": {"source": self.source, "frame": frame,
                                          "vehicles": vehicles},
                    "violations.json": events,
                    "speed_violations.json": speed_events,
                    "all_violations.json": all_events}
        for filename, data in payloads.items():
            target = self.directory / filename
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temporary.replace(target)
        for filename, rows, fields in [
            ("vehicles.csv", vehicles, ["vehicle_id", "vehicle_type", "vehicle_number", "plate_status",
                                       "ocr_confidence", "observations", "first_frame", "last_frame",
                                       "first_read_frame", "plate_image", "speed_kmh", "peak_speed_kmh",
                                       "speed_status", "speed_limit_kmh", "speed_calibration",
                                       "last_speed_kmh", "last_speed_age_seconds"]),
            ("violations.csv", events, ["vehicle_id", "vehicle_type", "vehicle_number", "plate_status",
                                        "ocr_confidence", "frame", "video_seconds", "signal", "violation_type", "event_type",
                                        "number_at_event", "number_at_crossing", "number_source", "plate_image", "evidence_image"]),
            ("speed_violations.csv", speed_events,
             ["vehicle_id", "vehicle_type", "vehicle_number", "estimated_speed_kmh", "speed_limit_kmh",
              "calibration_status", "violation_type", "event_type", "frame", "video_seconds", "signal", "plate_status",
              "ocr_confidence", "number_at_event", "number_source", "plate_image", "evidence_image"]),
            ("all_violations.csv", all_events,
             ["vehicle_id", "vehicle_type", "vehicle_number", "violation_type", "event_type",
              "frame", "video_seconds", "signal", "estimated_speed_kmh", "speed_limit_kmh",
              "calibration_status", "plate_status", "ocr_confidence", "number_at_event",
              "number_at_crossing", "number_source", "plate_image", "evidence_image"]),
        ]:
            with (self.directory / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
