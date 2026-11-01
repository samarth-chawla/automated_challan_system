"""A road tripwire aligned with the roadside prohibition sign in tr2.mp4."""

import cv2
import numpy as np

REFERENCE_SIGN_CENTER = (187.5, 185.0)  # measured on frame 98 at 1100 x 700


class RoadBoundary:
    def __init__(self, line, track_sign=True):
        self.reference_line = [tuple(point) for point in line]
        self.track_sign = track_sign
        self.offset = np.zeros(2, dtype=float)
        self.missed = 0
        self.reliable = not track_sign

    @staticmethod
    def find_sign(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red = (cv2.inRange(hsv, (0, 80, 60), (12, 255, 255)) |
               cv2.inRange(hsv, (168, 80, 60), (179, 255, 255)))
        search = np.zeros(red.shape, dtype=np.uint8)
        search[60:320, 100:320] = 255
        contours, _ = cv2.findContours(red & search, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if 40 < width < 110 and 50 < height < 140 and 0.55 < width / height < 1.3 and area > 1200:
                candidates.append((area, np.array([x + width / 2, y + height / 2])))
        return max(candidates, key=lambda value: value[0])[1] if candidates else None

    def update(self, frame):
        if self.track_sign:
            center = self.find_sign(frame)
            if center is not None:
                self.offset = center - REFERENCE_SIGN_CENTER
                self.missed = 0
            else:
                self.missed += 1
            self.reliable = self.missed <= 10 and (center is not None or self.reliable)
        return [(int(round(x + self.offset[0])), int(round(y + self.offset[1])))
                for x, y in self.reference_line]

    def to_reference(self, point):
        return tuple(np.asarray(point, dtype=float) - self.offset)


def resolve_line(options, default):
    if options.line is not None:
        x1, y1, x2, y2 = options.line
        return [(x1, y1), (x2, y2)], False
    if options.line_y is not None:
        return [(default[0][0], options.line_y), (default[1][0], options.line_y)], False
    return list(default), not options.fixed_line
