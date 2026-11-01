"""Measure vehicle movement from image features, independently of box resizing."""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Motion:
    moving: bool = False
    dx: float = 0.0
    dy: float = 0.0
    samples: int = 0
    points: int = 0

    @property
    def label(self):
        return "MOVING" if self.moving else ("STOPPED" if self.samples >= 4 else "CHECKING")


class VehicleMotion:
    """Sparse optical flow with occluder masking and background-motion removal.

    Bounding-box expansion does not contribute to measured movement. Too few
    reliable features produce unknown movement, never permission to flag a car.
    """

    def __init__(self, direction="down", window=10, max_gap=2):
        self.direction = direction
        self.window = window
        self.max_gap = max_gap
        self.previous_gray = None
        self.previous_vehicles = {}
        self.previous_frame = None
        self.history = {}
        self.estimates = {}

    @staticmethod
    def features(gray, mask, count=60):
        return cv2.goodFeaturesToTrack(gray, maxCorners=count, qualityLevel=0.02,
                                      minDistance=5, blockSize=5, mask=mask)

    @staticmethod
    def flow(previous, current, points):
        if points is None or len(points) < 4:
            return np.empty((0, 2), dtype=np.float32)
        following, ok, errors = cv2.calcOpticalFlowPyrLK(previous, current, points, None,
                                                        winSize=(21, 21), maxLevel=3)
        if following is None:
            return np.empty((0, 2), dtype=np.float32)
        backward, reverse_ok, _ = cv2.calcOpticalFlowPyrLK(current, previous, following, None,
                                                         winSize=(21, 21), maxLevel=3)
        if backward is None:
            return np.empty((0, 2), dtype=np.float32)
        valid = (ok.ravel() == 1) & (reverse_ok.ravel() == 1)
        valid &= np.linalg.norm(backward.reshape(-1, 2) - points.reshape(-1, 2), axis=1) < 1.0
        valid &= errors.ravel() < 25
        vectors = (following - points).reshape(-1, 2)
        vectors[~valid] = np.nan
        return vectors

    def update(self, frame, vehicles, frame_number):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        estimates = {vehicle_id: Motion() for vehicle_id in vehicles}
        consecutive = self.previous_frame is not None and frame_number - self.previous_frame <= self.max_gap
        if self.previous_gray is not None and consecutive:
            # Find corners once, then run one batched LK calculation for all IDs.
            # Rebuilding image pyramids per vehicle makes a CPU demo needlessly slow.
            corners = self.features(self.previous_gray, None, 1500)
            if corners is None:
                corners = np.empty((0, 1, 2), dtype=np.float32)
            coordinates = corners.reshape(-1, 2).astype(int)

            def select(mask, count):
                if not len(corners):
                    return corners
                inside = mask[coordinates[:, 1], coordinates[:, 0]] != 0
                return corners[inside][:count]

            background_mask = np.full(gray.shape, 255, dtype=np.uint8)
            for box in self.previous_vehicles.values():
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(background_mask, (x1, y1), (x2, y2), 0, -1)
            # Exclude the synthetic lamp overlay from camera-motion estimation.
            background_mask[:220, 980:] = 0
            groups = [(None, select(background_mask, 100))]
            for vehicle_id, box in vehicles.items():
                if vehicle_id not in self.previous_vehicles:
                    continue
                x1, y1, x2, y2 = self.previous_vehicles[vehicle_id]
                width, height = x2 - x1, y2 - y1
                mask = np.zeros(gray.shape, dtype=np.uint8)
                # Track the body/windows, not the changing box bottom or road.
                cv2.rectangle(mask, (int(x1 + width * 0.12), int(y1 + height * 0.10)),
                              (int(x2 - width * 0.12), int(y1 + height * 0.70)), 255, -1)
                # A nearer vehicle's features must not become the bus's motion.
                for other_id, other in self.previous_vehicles.items():
                    if other_id != vehicle_id and other[3] > y2 + 12:
                        ox1, oy1, ox2, oy2 = map(int, other)
                        cv2.rectangle(mask, (ox1 - 3, oy1 - 3), (ox2 + 3, oy2 + 3), 0, -1)
                groups.append((vehicle_id, select(mask, 60)))
            points = np.concatenate([group[1] for group in groups])
            combined_flow = self.flow(self.previous_gray, gray, points)
            grouped_flow = {}
            offset = 0
            for vehicle_id, group in groups:
                vectors = combined_flow[offset:offset + len(group)]
                grouped_flow[vehicle_id] = vectors[np.isfinite(vectors).all(axis=1)]
                offset += len(group)
            background_flow = grouped_flow.pop(None)
            camera = np.median(background_flow, axis=0) if len(background_flow) >= 8 else np.zeros(2)
            for vehicle_id, vectors in grouped_flow.items():
                history = self.history.setdefault(vehicle_id, deque(maxlen=self.window))
                while history and frame_number - history[0][0] > self.window:
                    history.popleft()
                if len(vectors) >= 6:
                    displacement = np.median(vectors, axis=0) - camera
                    if np.linalg.norm(displacement) < 30:  # reject flow jumps
                        history.append((frame_number, float(displacement[0]), float(displacement[1])))
                # Stale motion is never carried through an occlusion.
                if not history or history[-1][0] != frame_number:
                    continue
                dx = sum(value[1] for value in history)
                dy = sum(value[2] for value in history)
                signed_y = dy if self.direction == "down" else -dy
                # Cars approach diagonally in this view: most visible displacement
                # is horizontal, while their road-contact point moves downwards.
                moving = len(history) >= 4 and signed_y >= 1.5 and np.hypot(dx, dy) >= 6
                estimates[vehicle_id] = Motion(bool(moving), dx, dy, len(history), len(vectors))
        else:
            self.history.clear()
        # Tracker IDs are not reused; expire optical-flow state after tracks disappear.
        self.history = {key: value for key, value in self.history.items() if key in vehicles}
        self.previous_gray = gray
        self.previous_vehicles = {key: list(value) for key, value in vehicles.items()}
        self.previous_frame = frame_number
        self.estimates = estimates
        return estimates
