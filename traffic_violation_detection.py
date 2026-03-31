import cv2
import os
import time
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
VIDEO_PATH = "tr.mp4"          # CCTV or test video
MODEL_PATH = "yolov8m.pt"      # Model name (auto-downloads if missing)
CONF_THRESHOLD = 0.75
# ----------------------------------------

# 🧠 Automatically download YOLOv8 if not found
if not os.path.exists(MODEL_PATH):
    print(f"🔽 Model '{MODEL_PATH}' not found — downloading automatically...")
model = YOLO(MODEL_PATH)  # This triggers download if file is missing

# COCO class names from the model
coco = model.model.names
TargetLabels = ["bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]

# ---------------- COORDINATES ----------------
# Adjust these polygons for your camera view
RedLight = np.array([[998, 125], [998, 155], [972, 152], [970, 127]])
GreenLight = np.array([[971, 200], [996, 200], [1001, 228], [971, 230]])
ROI = np.array([[910, 372], [388, 365], [338, 428], [917, 441]])
# ---------------------------------------------

def is_region_light(image, polygon, brightness_threshold=128):
    """Check if a region (polygon) is bright enough to consider the light ON."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [np.array(polygon)], 255)
    roi = cv2.bitwise_and(gray, gray, mask=mask)
    mean_brightness = cv2.mean(roi, mask=mask)[0]
    return mean_brightness > brightness_threshold

def draw_text_with_background(frame, text, position, font, scale, text_color, background_color, border_color, thickness=2, padding=5):
    """Draw text with background and border on the frame."""
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x - padding, y - th - padding),
                  (x + tw + padding, y + base + padding),
                  background_color, cv2.FILLED)
    cv2.rectangle(frame, (x - padding, y - th - padding),
                  (x + tw + padding, y + base + padding),
                  border_color, thickness)
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, lineType=cv2.LINE_AA)

# ---------------- MAIN PROCESS ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Error: Could not open video '{VIDEO_PATH}'")
    exit()

print("✅ YOLOv8 Model loaded successfully. Starting detection...")
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("📹 Video ended or cannot read frames.")
        break

    frame = cv2.resize(frame, (1100, 700))
    frame_count += 1

    # Draw reference polygons
    cv2.polylines(frame, [RedLight], True, [0, 0, 255], 1)
    cv2.polylines(frame, [GreenLight], True, [0, 255, 0], 1)
    cv2.polylines(frame, [ROI], True, [255, 0, 0], 2)

    # Predict
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        classes = result.boxes.cls

        for box, conf, cls in zip(boxes, confs, classes):
            label = coco[int(cls)]
            if label not in TargetLabels:
                continue

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 255, 0], 2)
            draw_text_with_background(
                frame,
                f"{label.capitalize()} ({conf*100:.1f}%)",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (255, 255, 255),
                (0, 0, 0),
                (0, 0, 255)
            )

            # Check red light and ROI violation
            if is_region_light(frame, RedLight):
                # If any corner of bounding box inside ROI
                if (cv2.pointPolygonTest(ROI, (x1, y2), False) >= 0 or
                    cv2.pointPolygonTest(ROI, (x2, y2), False) >= 0):
                    draw_text_with_background(
                        frame,
                        f"🚨 {label.capitalize()} violated red signal!",
                        (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.8,
                        (255, 255, 255),
                        (0, 0, 0),
                        (0, 0, 255)
                    )
                    cv2.polylines(frame, [ROI], True, [0, 0, 255], 3)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 3)

                    timestamp = int(time.time())
                    filename = f"violation_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"[{frame_count}] 🚗 Violation saved → {filename}")

    cv2.imshow("Traffic Violation Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Process finished.")