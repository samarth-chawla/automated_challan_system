import cv2
import os
import time
import numpy as np
import easyocr
from ultralytics import YOLO

# ---------------- CONFIG ----------------
VIDEO_PATH = "tr.mp4"                  # Path to CCTV or sample video
VEHICLE_MODEL_PATH = "yolov8m.pt"      # Vehicle detection model
PLATE_MODEL_PATH = "yolov8x.pt"        # Plate detection model
CONF_THRESHOLD = 0.75
# ----------------------------------------

# 🧠 Auto-download YOLO models if missing
if not os.path.exists(VEHICLE_MODEL_PATH):
    print(f"🔽 Downloading {VEHICLE_MODEL_PATH} ...")
vehicle_model = YOLO(VEHICLE_MODEL_PATH)

if not os.path.exists(PLATE_MODEL_PATH):
    print(f"🔽 Downloading {PLATE_MODEL_PATH} ...")
plate_model = YOLO(PLATE_MODEL_PATH)

# Initialize OCR reader (EasyOCR)
ocr_reader = easyocr.Reader(['en'], gpu=False)

# COCO class names from YOLO
coco = vehicle_model.model.names
TargetLabels = ["bicycle", "car", "motorcycle", "bus", "truck"]

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
    """Draw readable text with background rectangle and border."""
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

print("✅ YOLOv8 Models loaded successfully. Starting detection...")
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("📹 Video ended or cannot read frames.")
        break

    frame = cv2.resize(frame, (1100, 700))
    frame_count += 1

    # Draw reference regions
    cv2.polylines(frame, [RedLight], True, [0, 0, 255], 1)
    cv2.polylines(frame, [GreenLight], True, [0, 255, 0], 1)
    cv2.polylines(frame, [ROI], True, [255, 0, 0], 2)

    # Predict vehicles
    results = vehicle_model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

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

            # --- Check for violation ---
            if is_region_light(frame, RedLight):  # Red light is ON
                # Vehicle crosses ROI
                if (cv2.pointPolygonTest(ROI, (x1, y2), False) >= 0 or
                    cv2.pointPolygonTest(ROI, (x2, y2), False) >= 0):

                    # Violation detected
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

                    # Crop vehicle area
                    vehicle_crop = frame[y1:y2, x1:x2]

                    # Detect license plate inside vehicle
                    lp_results = plate_model.predict(vehicle_crop, conf=0.5, verbose=False)

                    plate_text = "Unknown"
                    for lp_result in lp_results:
                        lp_boxes = lp_result.boxes.xyxy
                        for lp_box in lp_boxes:
                            px1, py1, px2, py2 = map(int, lp_box)
                            license_crop = vehicle_crop[py1:py2, px1:px2]

                            # OCR on plate
                            ocr_results = ocr_reader.readtext(license_crop)
                            if ocr_results:
                                plate_text = " ".join([res[1] for res in ocr_results])
                                print(f"[{frame_count}] 🔢 License Plate Detected: {plate_text}")

                            # Save cropped plate
                            timestamp = int(time.time())
                            lp_filename = f"plate_{timestamp}.jpg"
                            cv2.imwrite(lp_filename, license_crop)
                            print(f"[{frame_count}] 📸 License plate saved → {lp_filename}")

                    # Save violation frame (only when violated)
                    timestamp = int(time.time())
                    violation_filename = f"violation_{timestamp}.jpg"
                    cv2.imwrite(violation_filename, frame)
                    print(f"[{frame_count}] 🚗 Violation saved → {violation_filename} | Plate: {plate_text}")

    # --- Real GUI Window Display ---
    cv2.imshow("🚦 Traffic Violation Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        print("🛑 ESC pressed. Exiting...")
        break
    elif key == ord('p'):  # pause/play
        print("⏸ Paused. Press any key to continue...")
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
print("✅ Process finished.")
