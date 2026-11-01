"""Perspective-plane speed estimates using source-video time and explicit calibration."""

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SpeedReading:
    kmh: float | None = None
    status: str = "collecting"
    over_limit: bool = False
    new_alert: bool = False
    last_kmh: float | None = None
    last_age_seconds: float | None = None


def load_calibration(path, limit=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if limit is not None:
        data["speed_limit_kmh"] = limit
    if data.get("mode") not in {"demo", "calibrated"}:
        raise ValueError("Speed calibration mode must be demo or calibrated")
    points = np.asarray(data.get("image_points"), dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all() or not cv2.isContourConvex(points):
        raise ValueError("Four ordered corners of a convex road quadrilateral are required")
    if cv2.contourArea(points) < 100:
        raise ValueError("Speed calibration region is too small")
    for name in ["road_width_m", "road_length_m", "speed_limit_kmh"]:
        if not isinstance(data.get(name), (int, float)) or not 0 < data[name] < 10000:
            raise ValueError(f"{name} must be a finite positive number")
    return data


class SpeedEstimator:
    def __init__(self, calibration, fps):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("A positive source-video FPS is required")
        self.calibration = calibration
        self.fps = fps
        self.limit = float(calibration["speed_limit_kmh"])
        self.demo = calibration["mode"] == "demo"
        self.polygon = np.asarray(calibration["image_points"], dtype=np.float32)
        width, length = calibration["road_width_m"], calibration["road_length_m"]
        world = np.float32([[0, 0], [width, 0], [width, length], [0, length]])
        self.transform = cv2.getPerspectiveTransform(self.polygon, world)
        self.history = {}
        self.smoothed = {}
        self.streaks = {}
        self.alerted = set()
        self.last_valid = {}
        self.window = max(12, round(fps * 0.75))
        self.min_span = max(4, round(fps * 0.35))

    def update(self, vehicle_id, point, frame, moving, reliable=True):
        history = self.history.setdefault(vehicle_id, deque(maxlen=self.window + 1))
        smooth = self.smoothed.setdefault(vehicle_id, deque(maxlen=3))

        def unavailable(status, clear=True):
            if clear:
                history.clear()
                smooth.clear()
            self.streaks[vehicle_id] = 0
            if status in {"tracking_jump", "invalid_timestamp", "alignment_unknown"}:
                self.last_valid.pop(vehicle_id, None)
            last = self.last_valid.get(vehicle_id)
            age = (frame - last[0]) / self.fps if last else None
            if last and 0 <= age <= 2.0:
                return SpeedReading(status=status, last_kmh=last[1], last_age_seconds=age)
            return SpeedReading(status=status)

        if not reliable:
            return unavailable("alignment_unknown")
        if cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False) < 0:
            return unavailable("outside_calibration")
        world = cv2.perspectiveTransform(np.float32([[point]]), self.transform)[0, 0].astype(float)
        if not np.isfinite(world).all():
            return unavailable("projection_invalid")
        if history and frame - history[-1][0] > 3:
            history.clear()
            smooth.clear()
        if history and frame <= history[-1][0]:
            return unavailable("invalid_timestamp")
        if history:
            elapsed = (frame - history[-1][0]) / self.fps
            step_speed = np.linalg.norm(world - history[-1][1]) / elapsed * 3.6
            if step_speed > 180:
                return unavailable("tracking_jump")
        history.append((frame, world))
        while history and frame - history[0][0] > self.window:
            history.popleft()
        if not moving:
            return unavailable("motion_unconfirmed", clear=False)
        if len(history) < 8 or frame - history[0][0] < self.min_span:
            return unavailable("collecting", clear=False)

        times = np.array([(value[0] - history[0][0]) / self.fps for value in history])
        positions = np.array([value[1] for value in history])
        # Median pairwise slopes are robust to a few changing-box/occlusion samples.
        slopes = [(positions[j] - positions[i]) / (times[j] - times[i])
                  for i in range(len(times)) for j in range(i + 1, len(times))
                  if times[j] - times[i] >= 0.1]
        velocity = np.median(slopes, axis=0)
        intercept = np.median(positions - times[:, None] * velocity, axis=0)
        residuals = np.linalg.norm(positions - (intercept + times[:, None] * velocity), axis=1)
        tolerance = max(0.15, 3 * float(np.median(residuals)))
        valid = residuals <= tolerance
        if valid.sum() < 8 or valid.mean() < 0.65 or times[valid][-1] - times[valid][0] < 0.3:
            return unavailable("noisy_track", clear=False)
        velocity = np.polyfit(times[valid], positions[valid], 1)[0]
        value = float(np.linalg.norm(velocity) * 3.6)
        if not np.isfinite(value) or value > 180:
            return unavailable("unreliable_estimate")
        smooth.append(value)
        value = float(np.median(smooth))
        self.last_valid[vehicle_id] = (frame, value)
        self.streaks[vehicle_id] = self.streaks.get(vehicle_id, 0) + 1 if value > self.limit else 0
        over = self.streaks[vehicle_id] >= 3
        new_alert = over and vehicle_id not in self.alerted
        if new_alert:
            self.alerted.add(vehicle_id)
        return SpeedReading(value, "demo_estimate" if self.demo else "calibrated_estimate", over, new_alert)
