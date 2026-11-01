"""Readable two-line labels, kept inside the video and away from each other."""

import cv2

STATUS_TEXT = {
    "collecting": "Speed: collecting frames",
    "motion_unconfirmed": "Speed: motion uncertain",
    "outside_calibration": "Speed: outside road zone",
    "alignment_unknown": "Speed: road alignment lost",
    "projection_invalid": "Speed: projection uncertain",
    "invalid_timestamp": "Speed: tracking reset",
    "tracking_jump": "Speed: tracking reset",
    "noisy_track": "Speed: track uncertain",
    "unreliable_estimate": "Speed: estimate uncertain",
    "disabled": "Speed: disabled",
}


def speed_caption(estimate):
    if estimate is None:
        return "Speed: collecting frames", "WAIT"
    if estimate.kmh is not None:
        return f"Est speed: {estimate.kmh:.1f} km/h", f"{estimate.kmh:.1f}"
    if estimate.last_kmh is not None:
        return f"Last: {estimate.last_kmh:.1f} km/h ({estimate.last_age_seconds:.1f}s ago)", f"L{estimate.last_kmh:.1f}"
    brief = {"outside_calibration": "ZONE", "motion_unconfirmed": "MOTION",
             "collecting": "WAIT", "tracking_jump": "RESET", "alignment_unknown": "ALIGN"}
    return STATUS_TEXT.get(estimate.status, "Speed: tracking uncertain"), brief.get(estimate.status, "CHECK")


def overlap(first, second):
    x1, y1, x2, y2 = first
    a1, b1, a2, b2 = second
    return max(0, min(x2, a2)-max(x1, a1)) * max(0, min(y2, b2)-max(y1, b1))


def place_labels(items, width=1100, height=700):
    """Prefer the top of each box, then search nearby space for readable labels."""
    occupied = [(0, 0, 720, 82), (985, 0, 1100, 230)]
    placements = []
    for item in sorted(items, key=lambda value: (-value.get("priority", 0),
                                                -(value["box"][2]-value["box"][0])*(value["box"][3]-value["box"][1]))):
        x1, y1, x2, y2 = map(int, item["box"])
        sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0] for line in item["lines"]]
        label_width = min(410, max(sizes) + 14)
        label_height = len(item["lines"]) * 19 + 10
        anchor = (x1, y1-label_height-5)
        candidates = [(x1, y1-label_height-5), (x1+5, y1+6), (x2+5, y1),
                      (x1-label_width-5, y1), (x1, y2+5)]
        for dy in [0, -55, 55, -110, 110, -165, 165]:
            for dx in [0, -150, 150, -300, 300]:
                candidates.append((anchor[0]+dx, anchor[1]+dy))
        best = None
        for cx, cy in candidates:
            cx = max(4, min(width-label_width-4, cx))
            cy = max(84, min(height-label_height-4, cy))
            rectangle = (cx, cy, cx+label_width, cy+label_height)
            collisions = sum(overlap(rectangle, previous) for previous in occupied)
            distance = abs(cx-anchor[0]) + abs(cy-anchor[1])
            score = collisions * 1000 + distance
            if best is None or score < best[0]:
                best = score, rectangle
        rectangle = best[1]
        occupied.append((rectangle[0]-3, rectangle[1]-3, rectangle[2]+3, rectangle[3]+3))
        placements.append((item, rectangle))
    return placements


def draw_labels(image, items):
    placements = place_labels(items)
    for item, (x1, y1, x2, y2) in placements:
        bx1, by1, bx2, by2 = map(int, item["box"])
        color = item["color"]
        cv2.line(image, (min(max(bx1, x1), x2), min(max(by1, y1), y2)),
                 (max(0, bx1), max(0, by1)), color, 1)
    for item, (x1, y1, x2, y2) in placements:
        color = item["color"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (20, 25, 35), -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        for index, line in enumerate(item["lines"]):
            while cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0] > x2-x1-12:
                line = line[:-4] + "..." if len(line) > 4 else "..."
            text_color = color if index == 0 else (240, 240, 240)
            cv2.putText(image, line, (x1+6, y1+19+index*19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, text_color, 1, cv2.LINE_AA)
