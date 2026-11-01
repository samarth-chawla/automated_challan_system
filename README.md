# Traffic Signal and Number-Plate Review Demo

A local Python/OpenCV demo for `tr2.mp4`: track vehicles, read visible plates, cache numbers by vehicle ID, and record movement alerts during a red signal. Observed boundary crossings and movement first seen beyond the boundary are recorded separately.

## Run

```powershell
.\.venv\Scripts\python.exe -X utf8 traffic_violation_detection1.py
```

`traffic_violation.py` and `traffic_violation_detection.py` launch the same pipeline. All launchers default to `tr2.mp4`, the plate-specific OCR reader, and `--event-mode red-zone`.

Click the video window: **Space** or **p** pauses/resumes; **Esc** or **q** exits. Restart an already running script after a code update. Previously generated videos retain their earlier results.

The display includes vehicle IDs, cached plate text, motion state, confidence, a plate-crop close-up, and red alerts. `(?)` marks a single reading; `REPEAT` means repeated agreement. OCR scores are model estimates, not independently measured accuracy.

Each currently tracked vehicle has a separate readable label with its ID/plate and a speed line. Labels are placed inside the video, avoid one another and the signal graphic, and use connecting lines when moved away from a box. The side table lists currently tracked IDs rather than mixing in cars that have already disappeared.

Speed captions explicitly distinguish:

- `Est speed: 9.0 km/h`: a current estimate.
- `Last: 13.5 km/h (1.4s ago)`: the last valid estimate for this same ID, retained for at most 2 seconds after the current estimate becomes unavailable. It is display-only and cannot trigger speeding alerts.
- `Speed: collecting frames`, `motion uncertain`, `outside road zone`, or `tracking reset`: why a number cannot currently be calculated. The table uses `WAIT`, `MOTION`, `ZONE`, or `RESET`; `L13.5` denotes a historical value.

Numbers cannot be supplied reliably for every visible vehicle. Small or occluded vehicles, short tracks, uncertain optical flow and points outside the calibration region can lack an estimate. The application displays that reason instead of silently hiding the speed line or inventing km/h.

Speed estimates and speed-limit alerts are also available. The default road dimensions are assumptions, and the interface labels them `DEMO`. See the speed section and [review explanation](REVIEW_NOTES.md) before presenting km/h as a measurement.

Inactive lamp outlines are grey. A vehicle is red only for an alert recorded during the current red phase. Earlier events turn amber and are explicitly labelled `PAST` with their recorded time; the history table is cumulative. Green/unknown signals cannot create a new red event, and event storage rejects them.

Yellow is recognized explicitly, including the yellow lamp shown at frame 215. Yellow frames do not create red-light events. A speed alert remains independent of signal color.

For a smooth review recording, process the video first:

```powershell
.\.venv\Scripts\python.exe -X utf8 traffic_violation_detection1.py --headless --save-video
```

Then play `runs/session_.../annotated.mp4`. Processing on CPU is slower than video playback; the recording is encoded at the source frame rate.

## Plate recognition and vehicle memory

1. YOLO detects vehicles; ByteTrack assigns session-local IDs.
2. `license_plate.pt` locates plates on the original video pixels. Each plate is associated with one containing vehicle.
3. A dedicated [FastPlateOCR CCT recognizer](https://github.com/ankandrew/fast-plate-ocr) reads the plate crops locally. RGB input, the matching model configuration, and model-specific preprocessing are used. No vehicle registration numbers are hard-coded in application code.
4. Readings are collected every 4 frames, independently of the signal. Confidence, repeated observations, and crop size weight the best current number for each ID. Repeated readings are refreshed less frequently.
5. Alerts reuse that ID's cached number. If the correct number becomes readable later, the same ID's alert is enriched, with provenance showing when that reading became available.

The earlier general-purpose EasyOCR reader misread the visible BMW as `B645`/`WB645`. The plate-specific reader correctly reads `YB6433` from those saved image crops; a real crop is included in `tests/fixtures/bmw_plate.png` for regression testing. Other blurred or occluded plates can still be missed or misread.

Use `--ocr-engine easyocr` only to compare the older reader. `yolov8x.pt` is no longer used as a supposed plate model.

## Boundary, signal alignment, and alert meaning

Settings are in `anpr_core.py`, in **1100 x 700 video coordinates**:

```text
Road line in reference frame 98: (190, 635) to (1080, 620)
Roadside sign center in reference frame 98: (187.5, 185)
Red lamp ellipse: center (1038, 55), horizontal/vertical radii (17, 22)
Green lamp ellipse: center (1038, 166), horizontal/vertical radii (17, 22)
Yellow lamp ellipse: center (1038, 111), horizontal/vertical radii (17, 22)
```

Measured lamp centers from the native frames are approximately `(1037.3, 54.8)` for red and `(1037.0, 165.9)` for green. The ellipses cover the full lamp rather than the offset rectangles used previously. Signal colors are measured before drawing any annotations.

The line is aligned approximately with the roadside sign pole's base, sloping gently across the road. `road_boundary.py` locates the prohibition sign's red rim and translates the line as the camera view moves. Crossing calculations use sign-stabilized coordinates and the full sloping segment. An extended loss of the sign disables event generation until geometry is reliable again. This is a manually calibrated demo tripwire, not automatic discovery of a painted zebra crossing.

The previous fixed boundary `(220, 420)`–`(880, 420)` remains in `ORIGINAL_ROI_line` for comparison. To supply a fixed line or stop automatic alignment:

```powershell
.\.venv\Scripts\python.exe traffic_violation_detection1.py --line 220 420 880 420
.\.venv\Scripts\python.exe traffic_violation_detection1.py --fixed-line
```

Two event types are kept distinct:

- **`demo_red_line_crossing` / CROSS:** a tracked ground point is observed moving from one side of the selected segment to the other during red, with independent motion evidence.
- **`demo_red_zone_movement` / ZONE:** the vehicle is already beyond the boundary when seen, but shows sustained movement in the monitored road zone during red. Its crossing was not captured. This is a review alert, not proof of a red-light crossing.

The default review mode records both kinds and displays separate counts for red alerts and captured crossings. It requires three additional motion-positive frames after the independent optical-flow motion check, excludes stationary vehicles, and deduplicates each vehicle ID. This fixes the missed yellow Audi `NN773`: its confirmed red-motion interval is frames 391-395, followed by green at 396. The zone alert is now created at frame 393. It does not increase an observed-crossing count just because a moving car is visible under red.

To require an actual observed crossing:

```powershell
.\.venv\Scripts\python.exe -X utf8 traffic_violation_detection1.py --event-mode crossing
```

The original strict count was low because many cars were first detected already beyond the old line. Sparse optical flow follows vehicle-body features, masks nearer occluders, and removes estimated camera motion. Box expansion alone cannot create an event. The previously false-positive stationary double-decker is included in the regression checks.

`tr2.mp4` contains an overlaid signal graphic and does not establish a verified physical stop line. Counts are therefore demo alerts; they should not be presented as validated traffic offences. Plate OCR, tracking continuity, and movement estimates also need visual checking.

## Saved files

Data is stored locally on disk, with no database or cloud storage. The default location is `runs/` inside this project (`D:\Desktop\College\Mini Project\traffic-signal-violation-detection\runs`). Use `--output-dir path` to choose another parent folder. The console prints the exact session directory at startup and exit.

Each run starts a fresh in-memory ID/number cache and creates a unique directory. JSON/CSV snapshots are written on the first processed frame, every 25 frames, when a new alert is created, and on normal exit:

```text
runs/session_YYYYMMDD_HHMMSS_microseconds/
  vehicle_cache.json   # Current number readings by ID
  vehicles.csv         # Seen IDs, best numbers, scores, frame ranges, crop paths
  violations.json      # Red alerts with explicit event_type and number provenance
  violations.csv       # The same typed alerts in table form
  speed_violations.json # Speed alerts, including estimate/limit/calibration status
  speed_violations.csv  # The same speed alerts in table form
  all_violations.json  # All red and speed alerts, ordered by source frame
  all_violations.csv   # Combined table, including violation_type and event_type
  speed_calibration.json # Exact calibration and threshold used for this session
  summary.json         # Models, geometry, processed frames and separate counts
  plates/              # Accepted crop evidence, named with ID/text/frame
  violations/          # Annotated alert images, named with ID/frame
  speed_violations/    # Annotated speed-alert images
  annotated.mp4        # When --save-video is supplied
  tracks.jsonl         # When --diagnostics is supplied
```

Every new alert stores `violation_type` in both JSON and CSV. Open `all_violations.csv` to review every category together. The detailed `event_type` preserves the distinction between demo alerts and calibrated speed estimates:

| `violation_type` | Meaning | `event_type` |
| --- | --- | --- |
| `red_light_crossing` | Observed crossing during red | `demo_red_line_crossing` |
| `red_zone_movement` | Motion beyond the boundary during red; crossing unseen | `demo_red_zone_movement` |
| `speeding` | Sustained estimate above the configured threshold | `demo_speed_limit_exceeded` or `estimated_speed_limit_exceeded` |

These categories retain the demo limitations described above. A vehicle can have both a red alert and a speeding alert, and both rows are retained. `vehicle_type` separately identifies the vehicle class, such as car, bus or truck.

Other fields include `vehicle_id`, `vehicle_number`, `number_at_event`, `number_at_crossing`, `number_source`, `plate_status`, `ocr_confidence`, `frame`, `video_seconds`, `plate_image`, and `evidence_image`. Image paths are relative to their session folder.

`number_at_event` preserves the original reading. `number_at_crossing` is only populated for an observed crossing. Later corrected readings update `vehicle_number`; `number_source` identifies reads before/at/after the event or crossing. Unread values stay empty. `single_read` and `repeated_read` describe support, not guaranteed correctness.

IDs apply to one run, not to a vehicle's chassis identity. Occlusion or detector failures can split a vehicle into multiple track IDs. Disk snapshots remain until you remove the session directory; another run does not reuse them to identify vehicles.

Previous sessions retain their original export schema. Loose screenshots from the earlier script are preserved in `runs/legacy_evidence/`; they have no reliable recorded violation type and are not reclassified.

`.gitignore` excludes generated session data, model weights, local input videos, environments, logs and caches. Source code, calibration JSON and test fixtures remain eligible for version control.

## Speed estimates and speed alerts

`speed_calibration.demo.json` explicitly assumes a 9 m wide by 20 m long flat road patch. Four sign-stabilized image points map to that rectangle using a perspective transform. Source-frame timestamps and robust position fits over about 0.75 seconds produce estimated km/h. Tracking jumps, unreliable alignment, insufficient motion, and points outside the calibrated patch produce no speed value rather than an invented number.

The default comparison limit is 50 km/h. These dimensions and this limit are demonstration settings, not measured road dimensions or a verified posted speed limit. The UI, session calibration file, and exported speed events retain the `demo` status. For an illustrative low-threshold review run:

```powershell
.\.venv\Scripts\python.exe -X utf8 traffic_violation_detection1.py --speed-limit 15
```

The lower threshold exercises the alert flow; it is not a claim that 15 km/h is the legal limit. Sustained over-limit estimates create one separate speed event per vehicle ID. Speed alerts can occur on green or red, and one ID may have both red and speed events without either record replacing the other. Both records use the same cached plate and can be enriched by later OCR.

Vehicle records include the latest/peak estimate, speed status, threshold and calibration mode. Speed events include estimated km/h, the comparison limit, source frame/time, current signal, number provenance, and evidence image.

For real km/h validation, measure the road patch/camera geometry, enter the actual dimensions and posted limit in a separate configuration, and check against a reliable reference. Set its mode to `calibrated` only when supplying measured dimensions; the result remains an estimate that needs validation. Use `--speed-config path.json` for that file or `--no-speed` to disable speed estimation.

See [Ultralytics' calibration guidance](https://docs.ultralytics.com/guides/speed-estimation/) and [OpenCV perspective transforms](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html).

## Run after cloning from GitHub (Windows PowerShell)

Prerequisites: Git, 64-bit Python 3.12 with the `py` launcher, and internet access for dependency/model downloads. These commands use the environment's Python directly, so activating it or changing PowerShell's execution policy is unnecessary.

1. Clone the repository and enter its folder:

   ```powershell
   git clone https://github.com/samarth-chawla/automated_challan_system.git
   cd automated_challan_system
   ```

2. Create the environment and install dependencies:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -X utf8 -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -X utf8 -m pip install --force-reinstall --no-deps opencv-python
   ```

   Install the desktop OpenCV files last because OCR dependencies also install headless OpenCV. The desktop build is needed for the video window. If `py -3.12` is unavailable, install 64-bit Python 3.12 before continuing.

3. Download the plate detector and OCR assets:

   ```powershell
   .\.venv\Scripts\python.exe -X utf8 download_plate_model.py
   ```

   This creates `license_plate.pt` and the OCR files in `models/`, verifying their SHA256 checksums. The vehicle detector `yolov8m.pt` downloads on first run if absent; keep internet available for that run.

4. Obtain the demo video `tr2.mp4` from the project owner and put it in the repository root, beside `anpr_demo.py`. Videos, model weights, `.venv/` and run results are excluded from Git, so cloning does not supply them. The configured signal and road geometry are specific to this demo clip; renaming another video to `tr2.mp4` does not calibrate it.

5. Start the application:

   ```powershell
   .\.venv\Scripts\python.exe -X utf8 traffic_violation_detection1.py
   ```

   Click the video window. **Space** or **p** pauses/resumes; **q** or **Esc** exits and saves the session. No API key or database setup is required.

6. Open the newly printed `runs/session_.../` directory. `all_violations.csv` and `all_violations.json` contain the combined typed alerts; `vehicles.csv` contains tracked vehicles and plate readings. The evidence images are in `violations/` and `speed_violations/`.

To process without a window and save an annotated video:

```powershell
.\.venv\Scripts\python.exe -X utf8 traffic_violation_detection1.py --headless --save-video
```

For a short initial check:

```powershell
.\.venv\Scripts\python.exe -X utf8 traffic_violation_detection1.py --headless --max-frames 30
```

After setup, the default inference runs locally. A GPU is optional; the default device is CPU.

## Update an existing clone

From the repository folder, after saving any local work:

```powershell
git pull
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 -m pip install --force-reinstall --no-deps opencv-python
.\.venv\Scripts\python.exe -X utf8 download_plate_model.py
.\.venv\Scripts\python.exe -X utf8 traffic_violation_detection1.py
```

The model downloader reuses verified files. Local videos and earlier session results remain on disk when pulling code updates.

## Models and OpenCV troubleshooting

Models:

- [Koushim dedicated plate detector](https://huggingface.co/Koushim/yolov8-license-plate-detection), pinned weights revision and SHA256.
- [FastPlateOCR CCT-S v2 global recognizer](https://github.com/ankandrew/fast-plate-ocr), local `models/cct_s_v2_global.onnx` and matching config, checked against expected SHA256 values.

Both OCR libraries depend on headless OpenCV, while the window needs desktop OpenCV. If `cv2.imshow` is unavailable, reinstall the desktop files last:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps opencv-python
```

On Windows, `cv2.getBuildInformation()` should report `GUI: WIN32UI` for a GUI-capable build.

## Options and tests

```powershell
# Short check around the BMW/red-signal section
.\.venv\Scripts\python.exe -X utf8 traffic_violation_detection1.py --headless --start-frame 280 --max-frames 110
# All options
.\.venv\Scripts\python.exe traffic_violation_detection1.py --help
# Logic, motion, cache and real plate-crop regression tests
.\.venv\Scripts\python.exe -m unittest -v test_anpr_core test_plate_reader test_speed_estimation test_detection_overlay
```

Other options include `--video`, `--vehicle-model`, `--plate-model`, `--device`, `--confidence`, `--ocr-every`, `--output-dir`, `--line`, `--line-y`, `--fixed-line`, `--direction`, `--plate-pattern`, `--diagnostics`, `--no-ocr`, `--speed-config`, `--speed-limit`, and `--no-speed`. Explicit `--line` or `--line-y` overrides use a fixed line. The default plate-format filter accepts 4-6-character Hong Kong-style numbers for this clip; change it for other footage.

`traffic_violation_detection.py` is a compatibility launcher for the current pipeline, so it also saves typed JSON/CSV records inside a session directory. It now defaults to `tr2.mp4`; the old detection-only screenshot loop has been removed. Other input videos require their own geometry and signal calibration.

Regression coverage includes the actual green-lamp crop from frame 98, neutral inactive-lamp outlines, amber historical alerts during green, rejection of green-event storage, diagonal road-line intersections, sign/camera alignment, vehicle motion, and the real BMW plate crop. Other plate values still need visual verification.

Regression coverage now also checks speed-history expiry, display-only historical estimates, label bounds and collisions for the actual frame-215 boxes, and the real yellow-lamp crop. The yellow Audi `NN773` is recorded at frame 393 during red, before green at 396. These are estimated/demo alerts rather than validated traffic offences.

Verified on 1 November 2026: all 47 tests pass and the full 715-frame recording decodes. Frame 215 is explicitly `YELLOW`, with all seven tracked vehicles labelled. Taxi `VN4712` shows `Last: 13.5 km/h (1.4s ago)` while outside the current calibration region. The run retains 6 red-zone events and 4 demo speed events with the 15 km/h review threshold; display-only historical speed values do not add alerts.
