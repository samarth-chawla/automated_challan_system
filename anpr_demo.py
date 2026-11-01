"""Tracked plate-recognition demo for tr2.mp4. See README for calibration limits."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from anpr_core import (FRAME_SIZE, GreenLight, YellowLight, PLATE_PATTERN, RedEventMonitor,
                       ROI_line, RedLight, SessionStore, normalize_plate,
                       event_display_state, plate_owner, signal_state)
from road_boundary import RoadBoundary, resolve_line
from vehicle_motion import VehicleMotion
from speed_estimation import SpeedEstimator, SpeedReading, load_calibration
from detection_overlay import draw_labels, speed_caption

ROOT = Path(__file__).resolve().parent
WINDOW_NAME = "Vehicle IDs, Number Plates and Red-Light Events"
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "tr2.mp4")
    parser.add_argument("--vehicle-model", default=str(ROOT / "yolov8m.pt"))
    parser.add_argument("--plate-model", type=Path, default=ROOT / "license_plate.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs",
                        help="Parent directory for a new session containing JSON/CSV records and evidence")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes the whole video")
    parser.add_argument("--start-frame", type=int, default=0, help="Zero-based start frame for testing")
    parser.add_argument("--ocr-every", type=int, default=4, help="Frames between plate-reading passes")
    parser.add_argument("--confidence", type=float, default=0.1, help="Low-score detections help ByteTrack keep IDs")
    parser.add_argument("--device", default="cpu", help="cpu or an Ultralytics device such as 0")
    parser.add_argument("--line-y", type=int, default=None, help="Override with a fixed horizontal road line")
    parser.add_argument("--line", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Override both endpoints of a fixed line")
    parser.add_argument("--fixed-line", action="store_true", help="Disable sign-based camera alignment")
    parser.add_argument("--direction", choices=["down", "up"], default="down")
    parser.add_argument("--plate-pattern", default=PLATE_PATTERN, help="Full-match OCR filter for this camera")
    parser.add_argument("--diagnostics", action="store_true", help="Save per-frame boxes and measured motion")
    parser.add_argument("--no-ocr", action="store_true", help="Check tracking/motion without loading plate OCR")
    parser.add_argument("--ocr-engine", choices=["plate", "easyocr"], default="plate")
    parser.add_argument("--event-mode", choices=["red-zone", "crossing"], default="red-zone",
                        help="Separate red-zone movement alerts from captured crossings")
    parser.add_argument("--speed-config", type=Path, default=ROOT / "speed_calibration.demo.json")
    parser.add_argument("--speed-limit", type=float, default=None, help="Override the configured km/h limit")
    parser.add_argument("--no-speed", action="store_true", help="Disable speed estimation and alerts")
    opts = parser.parse_args(argv)
    if opts.ocr_every < 1 or opts.max_frames < 0 or opts.start_frame < 0:
        parser.error("ocr-every must be positive; max-frames/start-frame cannot be negative")
    if not 0 < opts.confidence <= 1 or (opts.line_y is not None and not 0 <= opts.line_y < FRAME_SIZE[1]):
        parser.error("confidence must be in (0,1]; line-y must be inside the resized image")
    if opts.line is not None:
        x1, y1, x2, y2 = opts.line
        if not (0 <= x1 < x2 < 1100 and 0 <= y1 < 700 and 0 <= y2 < 700):
            parser.error("line endpoints must be inside the video and ordered left to right")
    # Validate before loading expensive models.
    import re
    re.compile(opts.plate_pattern)
    return opts


def read_plate(reader, crop, pattern):
    """Try contrast variants, returning the strongest valid reading for this frame."""
    if hasattr(reader, "read_plate"):
        return reader.read_plate(crop, pattern)
    if crop.size == 0 or min(crop.shape[:2]) < 6:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height = max(48, round(gray.shape[0] * 320 / gray.shape[1]))
    gray = cv2.resize(gray, (320, height), interpolation=cv2.INTER_CUBIC)
    variants = [gray, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray),
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]]
    candidates = []
    for image in variants:
        # The plate detector already located a single text line. For stacked plates,
        # EasyOCR's text detector separates the rows before recognition.
        if crop.shape[1] / crop.shape[0] >= 2.4:
            reads = reader.recognize(image, allowlist=ALLOWLIST, detail=1, decoder="beamsearch")
        else:
            reads = reader.readtext(image, allowlist=ALLOWLIST, detail=1, decoder="beamsearch")
        if not reads:
            continue
        reads = sorted(reads, key=lambda item: (sum(p[1] for p in item[0]) / 4,
                                              sum(p[0] for p in item[0]) / 4))
        text = normalize_plate("".join(item[1] for item in reads), pattern)
        confidence = float(np.mean([item[2] for item in reads]))
        if text and confidence >= 0.35:
            candidates.append((text, confidence))
    return max(candidates, key=lambda item: item[1]) if candidates else None


def collect_plates(source, vehicles, store, detector, reader, frame_number, opts):
    """Detect on native pixels, associate to current IDs, then OCR before event checks."""
    result = detector.predict(source, conf=0.3, imgsz=960, verbose=False, device=opts.device)[0]
    sx, sy = FRAME_SIZE[0] / source.shape[1], FRAME_SIZE[1] / source.shape[0]
    assigned = {}
    for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
        x1, y1, x2, y2 = box
        display_box = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        owner = plate_owner(display_box, vehicles)
        if owner is not None and (owner not in assigned or score > assigned[owner][0]):
            assigned[owner] = (score, box, display_box)
    for vehicle_id, (_, box, _) in assigned.items():
        record = store.vehicles[vehicle_id]
        interval = opts.ocr_every * (3 if record.plate["plate_status"] == "repeated_read" else 1)
        if frame_number - record.last_ocr_frame < interval:
            continue
        record.last_ocr_frame = frame_number
        x1, y1, x2, y2 = map(int, box)
        pad = 2
        crop = source[max(0, y1 - pad):min(source.shape[0], y2 + pad),
                      max(0, x1 - pad):min(source.shape[1], x2 + pad)].copy()
        candidate = read_plate(reader, crop, opts.plate_pattern)
        if candidate is None:
            continue
        text, confidence = candidate
        previous_number = record.number
        filename = f"plates/vehicle_{vehicle_id}_{text}_frame_{frame_number}.jpg"
        quality = min(5.0, max(0.5, np.sqrt(crop.shape[0] * crop.shape[1]) / 30))
        if record.observe_plate(text, confidence, frame_number, filename, opts.plate_pattern, quality=quality):
            if not cv2.imwrite(str(store.directory / filename), crop):
                raise OSError(f"Could not save plate evidence: {filename}")
            store.refresh_event(record)
            if record.number != previous_number:
                print(f"PLATE frame={frame_number} ID={vehicle_id} number={record.number} "
                      f"confidence={record.plate['ocr_confidence']:.2f} "
                      f"status={record.plate['plate_status']}", flush=True)
    return {key: value[2] for key, value in assigned.items()}


def text(image, value, position, color=(235, 240, 245), scale=0.55):
    cv2.putText(image, value, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render(frame, store, vehicles, plate_boxes, signal, frame_number, fps, line, speed, motion,
           red_phase_start=0, speed_readings=None, speed_settings=None):
    speed_readings = speed_readings or {}
    speed_events = getattr(store, "speed_events", {})
    canvas = frame.copy()
    signal_color = {"RED": (50, 60, 245), "GREEN": (70, 220, 80), "YELLOW": (0, 220, 255)}.get(signal, (170, 170, 170))
    cv2.polylines(canvas, [RedLight], True, (40, 40, 240) if signal == "RED" else (140, 140, 140), 2)
    cv2.polylines(canvas, [GreenLight], True, (40, 230, 40) if signal == "GREEN" else (140, 140, 140), 2)
    cv2.polylines(canvas, [YellowLight], True, (0, 220, 255) if signal == "YELLOW" else (140, 140, 140), 2)
    cv2.line(canvas, tuple(line[0]), tuple(line[1]), signal_color, 2)
    cv2.rectangle(canvas, (0, 0), (720, 78), (20, 25, 35), -1)
    text(canvas, f"DEMO SIGNAL: {signal} | frame {frame_number} | video {(frame_number-1)/fps:.1f}s", (16, 29), signal_color, 0.65)
    text(canvas, "Motion alerts + observed crossings | Esc: exit | Space: pause", (16, 59), scale=0.53)
    labels = []
    for vehicle_id, box in vehicles.items():
        record = store.vehicles[vehicle_id]
        x1, y1, x2, y2 = map(int, box)
        event = store.events.get(vehicle_id)
        alert_state = event_display_state(event, signal, red_phase_start)
        estimate = speed_readings.get(vehicle_id)
        color = (30, 50, 245) if alert_state == "ACTIVE" else ((40, 190, 240) if alert_state == "PAST" else (50, 220, 130))
        if estimate and estimate.over_limit and alert_state != "ACTIVE":
            color = (230, 70, 220)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"#{vehicle_id} {record.number or 'PLATE READING'} | {motion[vehicle_id].label}"
        if record.plate["plate_status"] == "single_read":
            label += " (?)"
        lines = [label, speed_caption(estimate)[0]]
        alerts = []
        if estimate and estimate.over_limit:
            alerts.append("SPEED ALERT")
        if alert_state == "ACTIVE":
            alerts.append("RED CROSSING" if event["event_type"] == "demo_red_line_crossing" else "RED-ZONE ALERT")
        elif alert_state == "PAST":
            alerts.append(f"PAST RED @{event['video_seconds']:.1f}s")
        if alerts:
            lines.append(" | ".join(alerts))
        labels.append({"box": box, "lines": lines, "color": color,
                       "priority": 2 if alerts else (1 if record.number else 0)})
        cv2.circle(canvas, ((x1 + x2) // 2, min(699, y2)), 4, color, -1)
    for box in plate_boxes.values():
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 220, 255), 2)
    draw_labels(canvas, labels)

    panel = np.full((FRAME_SIZE[1], 410, 3), (28, 23, 20), dtype=np.uint8)
    title = "PLATES / SPEED DEMO" if speed_settings and speed_settings["mode"] == "demo" else "VEHICLE / PLATE MEMORY"
    text(panel, title, (18, 35), (80, 220, 250), 0.7)
    text(panel, f"Tracked now: {len(vehicles)} | Seen IDs: {len(store.vehicles)}", (18, 66), scale=0.52)
    known = sum(record.number is not None for record in store.vehicles.values())
    text(panel, f"Red history: {len(store.events)} | Speed alerts: {len(speed_events)}", (18, 92), scale=0.52)
    crossings = sum(event["event_type"] == "demo_red_line_crossing" for event in store.events.values())
    if speed_settings:
        prefix = "DEMO" if speed_settings["mode"] == "demo" else "ESTIMATE"
        text(panel, f"{prefix} limit: {speed_settings['speed_limit_kmh']:g} km/h | Plates: {known}", (18, 118), scale=0.50)
    else:
        text(panel, f"Captured crossings: {crossings} | Processing: {speed:.1f} fps", (18, 118), scale=0.47)
    text(panel, "ID     NUMBER       SPEED    STATE", (18, 155), (170, 175, 180), 0.47)
    recent = sorted((store.vehicles[key] for key in vehicles), key=lambda record: -(
        (record.box[2]-record.box[0])*(record.box[3]-record.box[1])))
    for index, record in enumerate(recent[:8]):
        plate = record.plate
        alert_state = event_display_state(store.events.get(record.vehicle_id), signal, red_phase_start)
        state = "RED NOW" if alert_state == "ACTIVE" else ("PAST" if alert_state == "PAST" else ("REPEAT" if plate["plate_status"] == "repeated_read" else "CHECK"))
        estimate = speed_readings.get(record.vehicle_id)
        if estimate and estimate.over_limit and alert_state != "ACTIVE":
            state = "SPEED"
        speed_value = speed_caption(estimate)[1]
        value = f"{record.vehicle_id:<6} {record.number or 'UNREAD':<11} {speed_value:<6} {state}"
        state_color = (80, 110, 255) if alert_state == "ACTIVE" else ((90, 200, 240) if alert_state == "PAST" else (220, 230, 235))
        text(panel, value, (18, 184 + index * 23), state_color, 0.48)
    featured = [store.vehicles[key] for key in vehicles if store.vehicles[key].number]
    if featured:
        record = max(featured, key=lambda value: (value.box[2]-value.box[0]) * (value.box[3]-value.box[1]))
        text(panel, f"PLATE CROP: #{record.vehicle_id} {record.number}", (18, 390), (80, 220, 250), 0.54)
        crop = cv2.imread(str(store.directory / record.plate["plate_image"]))
        if crop is not None:
            scale = min(350 / crop.shape[1], 70 / crop.shape[0])
            crop = cv2.resize(crop, (int(crop.shape[1]*scale), int(crop.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
            panel[405:405+crop.shape[0], 18:18+crop.shape[1]] = crop
    text(panel, "RECORDED ALERT HISTORY", (18, 505), (90, 200, 240), 0.55)
    history = sorted(list(store.events.values()) + list(speed_events.values()), key=lambda row: row["frame"])
    for index, event in enumerate(history[-4:]):
        kind = "SPEED" if "speed_limit" in event["event_type"] else ("CROSS" if event["event_type"] == "demo_red_line_crossing" else "ZONE")
        value = f"#{event['vehicle_id']} {event['vehicle_number'] or 'UNREAD'} {kind} @{event['video_seconds']:.1f}s"
        text(panel, value, (18, 532 + index * 24), scale=0.49)
    if speed_settings and speed_settings["mode"] == "demo":
        text(panel, f"Speed DEMO: assumed {speed_settings['road_width_m']:g} x {speed_settings['road_length_m']:g} m road.", (18, 647), (130, 200, 255), 0.46)
        text(panel, "Not measured speed / posted road limit.", (18, 671), scale=0.46)
    else:
        text(panel, "ZONE = movement; crossing not captured.", (18, 647), (130, 200, 255), 0.46)
        text(panel, "OCR and speed estimates need verification.", (18, 671), scale=0.46)
    return np.hstack([canvas, panel])


def main(argv=None):
    opts = arguments(argv)
    if not opts.video.is_file():
        raise SystemExit(f"Video not found: {opts.video}")
    if not opts.no_ocr and not opts.plate_model.is_file():
        raise SystemExit("Plate model missing. Run: .\\.venv\\Scripts\\python.exe download_plate_model.py")
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    vehicle_model = YOLO(opts.vehicle_model)
    plate_model = reader = None
    if not opts.no_ocr:
        plate_model = YOLO(str(opts.plate_model))
        if not any("plate" in str(name).lower() for name in plate_model.names.values()):
            raise SystemExit("The plate model has no plate class. Use license_plate.pt, not general COCO yolov8x.pt.")
        if opts.ocr_engine == "plate":
            from plate_reader import PlateReader
            reader = PlateReader()
            print("Loaded local plate-specific CCT OCR.", flush=True)
        else:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=opts.device != "cpu", verbose=False)
    cap = cv2.VideoCapture(str(opts.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {opts.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, opts.start_frame)
    session = opts.output_dir / datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
    store = SessionStore(session, fps, opts.video.resolve())
    speed_settings = None if opts.no_speed else load_calibration(opts.speed_config, opts.speed_limit)
    speed_estimator = SpeedEstimator(speed_settings, fps) if speed_settings else None
    if speed_settings:
        (session / "speed_calibration.json").write_text(json.dumps(speed_settings, indent=2), encoding="utf-8")
    line, track_sign = resolve_line(opts, ROI_line)
    boundary = RoadBoundary(line=line, track_sign=track_sign)
    monitor = RedEventMonitor(line=line, direction=opts.direction, mode=opts.event_mode)
    motion_estimator = VehicleMotion(direction=opts.direction)
    diagnostic_file = (session / "tracks.jsonl").open("w", encoding="utf-8") if opts.diagnostics else None
    vehicle_classes = [index for index, name in vehicle_model.names.items()
                       if name in {"car", "motorcycle", "bus", "truck"}]
    writer = None
    processed = 0
    frame_number = opts.start_frame
    started = time.perf_counter()
    previous_signal = None
    last_signal_change = frame_number
    print(f"SESSION {session}\nVideo: {opts.video}\nUsing ByteTrack IDs, optical-flow motion and line {line}.", flush=True)
    try:
        while True:
            ok, source = cap.read()
            if not ok:
                break
            frame_number += 1
            processed += 1
            frame = cv2.resize(source, FRAME_SIZE)
            signal = signal_state(frame)
            display_line = boundary.update(frame)
            if signal != previous_signal:
                print(f"SIGNAL frame={frame_number} state={signal}", flush=True)
                previous_signal = signal
                last_signal_change = frame_number
            result = vehicle_model.track(frame, persist=True, tracker="bytetrack.yaml",
                                         conf=opts.confidence, classes=vehicle_classes,
                                         verbose=False, device=opts.device)[0]
            vehicles = {}
            if result.boxes.id is not None:
                for box, vehicle_id, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                                result.boxes.id.int().cpu().tolist(),
                                                result.boxes.cls.int().cpu().tolist()):
                    vehicles[vehicle_id] = box
                    store.touch(vehicle_id, vehicle_model.names[cls], box, frame_number)
            plate_boxes = {}
            motion = motion_estimator.update(frame, vehicles, frame_number)
            if not opts.no_ocr and vehicles and (processed - 1) % opts.ocr_every == 0:
                plate_boxes = collect_plates(source, vehicles, store, plate_model, reader, frame_number, opts)
            new_events = []
            speed_readings = {key: SpeedReading(status="disabled") for key in vehicles} if not speed_estimator else {}
            for vehicle_id, (x1, y1, x2, y2) in vehicles.items():
                # Two red frames avoid a single-frame color glitch at a lamp transition.
                event_signal = signal if (boundary.reliable and (signal != "RED" or frame_number > last_signal_change)) else "UNKNOWN"
                point = boundary.to_reference(((x1 + x2) / 2, y2))
                if speed_estimator:
                    estimate = speed_estimator.update(vehicle_id, point, frame_number,
                                                      motion[vehicle_id].moving, reliable=boundary.reliable)
                    speed_readings[vehicle_id] = estimate
                    record = store.vehicles[vehicle_id]
                    record.speed_kmh = round(estimate.kmh, 1) if estimate.kmh is not None else None
                    record.speed_status = estimate.status
                    record.speed_limit_kmh = speed_estimator.limit
                    record.speed_calibration = speed_settings["mode"]
                    record.last_speed_kmh = round(estimate.last_kmh, 1) if estimate.last_kmh is not None else None
                    record.last_speed_age_seconds = round(estimate.last_age_seconds, 2) if estimate.last_age_seconds is not None else None
                    if estimate.kmh is not None:
                        record.peak_speed_kmh = round(max(record.peak_speed_kmh or 0, estimate.kmh), 1)
                    if estimate.new_alert:
                        event = store.register_speed_event(vehicle_id, frame_number, estimate.kmh,
                                                           speed_estimator.limit, speed_settings["mode"], signal)
                        new_events.append(event)
                        print(f"SPEED frame={frame_number} ID={vehicle_id} estimate={estimate.kmh:.1f} "
                              f"limit={speed_estimator.limit:g} calibration={speed_settings['mode']}", flush=True)
                event_type = monitor.observe(vehicle_id, point, frame_number, event_signal,
                                             moving=motion[vehicle_id].moving)
                if event_type and signal == "RED":
                    event = store.register_event(vehicle_id, frame_number, event_type=event_type, signal=signal)
                    new_events.append(event)
                    event["motion_dx"] = round(motion[vehicle_id].dx, 2)
                    event["motion_dy"] = round(motion[vehicle_id].dy, 2)
                    event["boundary_at_event"] = display_line
                    print(f"{event_type} frame={frame_number} ID={vehicle_id} "
                          f"cached_number={event['vehicle_number'] or 'UNREAD'}", flush=True)
            speed = processed / max(0.001, time.perf_counter() - started)
            display = render(frame, store, vehicles, plate_boxes, signal, frame_number, fps, display_line, speed, motion,
                             red_phase_start=last_signal_change, speed_readings=speed_readings,
                             speed_settings=speed_settings)
            if diagnostic_file:
                diagnostic_file.write(json.dumps({"frame": frame_number, "signal": signal,
                    "boundary": display_line, "boundary_reliable": boundary.reliable,
                    "vehicles": [{"id": key, "type": store.vehicles[key].vehicle_type, "box": box,
                                  "moving": motion[key].moving, "dx": round(motion[key].dx, 3),
                                  "dy": round(motion[key].dy, 3), "samples": motion[key].samples,
                                  "points": motion[key].points, "speed_kmh": store.vehicles[key].speed_kmh,
                                  "speed_status": store.vehicles[key].speed_status} for key, box in vehicles.items()]}) + "\n")
            for event in new_events:
                if not cv2.imwrite(str(session / event["evidence_image"]), display):
                    raise OSError("Could not save violation evidence")
            if new_events or processed % 25 == 0 or processed == 1:
                store.flush(frame_number)
                if diagnostic_file:
                    diagnostic_file.flush()
            if opts.save_video:
                if writer is None:
                    writer = cv2.VideoWriter(str(session / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (display.shape[1], display.shape[0]))
                    if not writer.isOpened():
                        raise OSError("Cannot create annotated.mp4")
                writer.write(display)
            if not opts.headless:
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break
                if key in (ord("p"), ord(" ")):
                    if cv2.waitKey(0) & 0xFF in (27, ord("q")):
                        break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if opts.max_frames and processed >= opts.max_frames:
                break
    except KeyboardInterrupt:
        print("Stopped; saving session memory.", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if diagnostic_file:
            diagnostic_file.close()
        if not opts.headless:
            cv2.destroyAllWindows()
        store.flush(frame_number)
        summary = {"video": str(opts.video.resolve()), "processed_frames": processed,
                   "vehicles": len(store.vehicles),
                   "vehicles_with_plate": sum(record.number is not None for record in store.vehicles.values()),
                   "red_alerts": len(store.events),
                   "red_line_events": sum(event["event_type"] == "demo_red_line_crossing" for event in store.events.values()),
                   "red_zone_alerts": sum(event["event_type"] == "demo_red_zone_movement" for event in store.events.values()),
                   "line": line, "direction": opts.direction, "event_mode": opts.event_mode,
                   "line_tracks_roadside_sign": track_sign,
                   "speed_alerts": len(store.speed_events), "speed_calibration": speed_settings,
                   "motion_required": True, "ocr_enabled": not opts.no_ocr,
                   "vehicle_model": opts.vehicle_model, "plate_model": str(opts.plate_model),
                   "note": "Demo signal overlay and manually placed line; not validated enforcement evidence."}
        (session / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved {processed} frames processed, {summary['vehicles_with_plate']} IDs with plate reads, "
              f"{len(store.events)} red alerts.\nResults: {session}", flush=True)
    return session


if __name__ == "__main__":
    main()
