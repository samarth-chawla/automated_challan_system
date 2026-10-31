# Traffic Signal Violation Detection System

Python desktop demo that uses Ultralytics YOLO and OpenCV to detect vehicles in a video, estimate whether a traffic light is on, and save frames flagged as red-signal violations. Output appears in an OpenCV window; there is no web application or backend server to start.

## Current status

Checked on 31 October 2026 in the local Windows checkout:

- The recommended entry point is `traffic_violation_detection.py`.
- `.venv` has Python 3.14.0, Ultralytics 8.4.154, OpenCV 5.0.0, PyTorch, NumPy, and EasyOCR installed. CPU inference succeeds; CUDA is unavailable in this environment.
- `tr.mp4`, `tr1.mp4`, `tr2.mp4`, `yolov8m.pt`, and `yolov8x.pt` are already present locally. All three videos can be opened and decoded.
- Both detection-only scripts passed a three-frame smoke check in `.venv`. Window display and image writes were suppressed; this does not validate full-video accuracy or the interactive GUI.
- The unused Python 3.10 `venv` environment was removed. Its OpenCV build reported `GUI: NONE`. Use the retained `.venv` for this checkout.
- `traffic_violation.py` adds experimental OCR. EasyOCR imports successfully, but the complete OCR pipeline has not been validated. Its configured `yolov8x.pt` has standard COCO object classes and **no license-plate class**, so it is not a dedicated plate detector.

Local environments, videos, model weights, and generated images are not part of the tracked source files. A fresh clone needs its own environment and input video.

## Run the existing checkout (Windows PowerShell)

```powershell
cd "D:\Desktop\College\Mini Project\traffic-signal-violation-detection"
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe traffic_violation_detection.py
```

Using the interpreter path directly avoids activating the wrong environment or PowerShell activation-policy issues. UTF-8 mode supports the scripts' emoji console messages.

- The script reads `tr.mp4` and loads `yolov8m.pt` from the current working directory.
- Frames are resized to 1100 x 700 and displayed with detection boxes and reference regions.
- Click the video window and press **Esc** to stop. Otherwise, processing stops at the end of the video.
- Flagged frames are saved as `violation_<timestamp>.jpg` in the project folder. There is no saved output video, database, or structured violation report.
- CPU processing may be slower than the source video's frame rate.

## Fresh installation

Use a 64-bit Python installation with compatible PyTorch wheels. Python 3.12 is a practical choice for a separate fresh environment; the existing local environment was checked with Python 3.14. The old README's Python 3.7 requirement does not match the current dependency constraints.

```powershell
git clone https://github.com/MohammedHamza0/traffic-signal-violation-detection.git
cd traffic-signal-violation-detection
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "opencv-python>=4.10.0" "ultralytics>=8.3.0" "numpy>=2.1.0" "setuptools>=75.0.0"
```

These packages are sufficient for the two detection-only scripts. Ultralytics installs PyTorch and its other dependencies. Use desktop OpenCV (`opencv-python`) because the scripts call `cv2.imshow`.

Before running:

1. Supply `tr.mp4` in the project folder, or edit `VIDEO_PATH` and calibrate the regions for your own video.
2. Supply `yolov8m.pt`. If missing, `YOLO("yolov8m.pt")` attempts to download the standard weights and needs internet access.
3. Run the existing-checkout commands above, using your own checkout path.

The videos and weights are ignored by Git; cloning does not supply them. Dependency versions in `requirements.txt` are lower bounds, so a new installation may resolve different versions from those checked above.

### Full dependencies, including experimental OCR

`requirements.txt` also includes EasyOCR:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

EasyOCR is needed only by `traffic_violation.py`. Its first reader initialization may download OCR weights. That script also loads `yolov8x.pt`, downloading it if missing.

EasyOCR depends on `opencv-python-headless`, while the video window needs `opencv-python`. These packages share the `cv2` module, and installing both can leave the active build without GUI support. The existing `.venv` has both installed but reports `GUI: WIN32UI`. The clean detection-only installation above avoids the EasyOCR dependency.

Installing OCR dependencies does not fix the plate detector. Reliable plate recognition still needs a model trained for license plates and corresponding class/crop handling in the script.

## Script variants

| Script | Default video | Behavior | Controls |
| --- | --- | --- | --- |
| `traffic_violation_detection.py` | `tr.mp4` | Recommended base demo; checks whether either bottom corner of a detected box lies inside a polygon while the red-light region is bright. | Esc exits |
| `traffic_violation_detection1.py` | `tr2.mp4` | Alternative geometry; flags boxes whose bottom edge is at or below a horizontal threshold while red is detected. | Esc exits |
| `traffic_violation.py` | `tr.mp4` | Experimental vehicle detection plus attempted plate detection and EasyOCR; also attempts to save `plate_<timestamp>.jpg`. | Esc exits; `p` pauses, any key resumes |

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe traffic_violation_detection1.py
# Experimental OCR variant, after installing its dependencies:
.\.venv\Scripts\python.exe traffic_violation.py
```

`tr1.mp4` is available locally but is not selected by any script by default. The scripts have no command-line options; edit configuration constants in the source.

## Configuration and limitations

Edit the selected script's `VIDEO_PATH`, model path, and `CONF_THRESHOLD` (currently `0.75`). Paths are resolved from the shell's current directory.

The base script and OCR variant use these coordinates **after resizing to 1100 x 700**:

```python
RedLight = np.array([[998, 125], [998, 155], [972, 152], [970, 127]])
GreenLight = np.array([[971, 200], [996, 200], [1001, 228], [971, 230]])
ROI = np.array([[910, 372], [388, 365], [338, 428], [917, 441]])
```

The alternative script has different light polygons and `ROI_line = [(220, 420), (880, 420)]`. For a different camera view, recalibrate the light regions and road region/line as well as changing the video filename.

This is a prototype with these accuracy and output limitations:

- Signal state uses mean grayscale brightness above `128` in a fixed polygon, rather than red/green color classification. Overlays are drawn before measurement and can affect it. The base script does not use green-light state to gate violations.
- Vehicles are evaluated independently per frame. There are no track IDs, direction checks, crossing history, or duplicate suppression; a vehicle already in a region can be flagged repeatedly.
- The line variant checks only `y2 >= roi_y`; it does not restrict vehicles to the horizontal span of the line or prove a crossing.
- Both detection-only scripts include `traffic light` among labels eligible for violation checks.
- Filenames use whole-second timestamps. Multiple saves in the same second can overwrite each other, and repeated detections can generate many images.
- There are no measured precision/recall results or validated plate-recognition results in this repository.

## Troubleshooting

- **Video cannot be opened:** run from the project folder, verify the file exists, and check `VIDEO_PATH`. Git does not include the videos.
- **`ModuleNotFoundError`:** install dependencies with the same interpreter used to run the script: `.\.venv\Scripts\python.exe -m pip ...`.
- **`UnicodeEncodeError`:** set `$env:PYTHONUTF8 = "1"` before launching Python.
- **`cv2.imshow` is not implemented:** inspect the active GUI backend:

  ```powershell
  .\.venv\Scripts\python.exe -c "import cv2; print(cv2.__version__); print(cv2.getBuildInformation())"
  ```

  On Windows, look for `GUI: WIN32UI`. `GUI: NONE` indicates a headless build. Use the checked `.venv`, or create a clean detection-only environment with the installation steps above.

- **Slow processing:** the checked environment uses CPU inference. A smaller compatible vehicle model can reduce processing cost if you update the model path; accuracy may change.
- **Missing or inaccurate violations:** verify the regions against the resized video and review the brightness and per-frame rules above. A successful run does not establish detection accuracy.

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)

Original project contact: [Mohammed Hamza Moawad Khalifa](https://www.linkedin.com/in/mohammed-hamza-moawad-khalifa/).
