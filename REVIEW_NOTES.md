# What to explain during the mini-project review

## Short explanation

"The system tracks vehicles by ID and keeps the best available number-plate reading for each ID. It measures the signal before drawing annotations and combines the road region with independent motion evidence. The yellow Audi NN773 was previously missed because the extra alert confirmation needed six red-motion frames but only five were available before green. The filter now requires three additional confirmed-motion frames, after the optical-flow check, and records its red-zone passage before the signal changes. Its actual initial crossing was not captured, so that event is labelled a zone alert."

## Speed estimation

"Speed is a prototype estimate, not a measured enforcement value. I map tracked road-contact points into a top-down road plane using a perspective transform. For this review, the road patch is assumed to be 9 metres wide and 20 metres long because its physical dimensions are unknown. I use displacement in that plane and timestamps derived from the source-video frame rate, not the computer's processing time. Speed is calculated as distance divided by elapsed time and converted to km/h by multiplying by 3.6."

For example, a hypothetical displacement of 2 m in 0.5 seconds is 14.4 km/h. That example illustrates the formula; it is not a measured speed from this footage.

The speed window covers about 0.75 seconds, needs enough source frames, and filters tracking jumps and noisy positions. Alerts require repeated over-limit estimates. Speed alerts are independent of the traffic signal, so they can legitimately appear on green; they are displayed separately from red alerts.

If asked why not every vehicle has a speed: "Every tracked vehicle has a visible ID and speed-status label, but a numeric estimate requires enough reliable movement inside the calibrated road patch. When that is unavailable, the application says whether it is collecting frames, uncertain, or outside the zone. It can briefly show a previous value labelled LAST with its age; that old value is never used to create a speeding alert."

## Demonstration assumptions

- `speed_calibration.demo.json` contains the assumed 9 m x 20 m road rectangle and a default illustrative 50 km/h comparison limit.
- The review recording uses `--speed-limit 15` to exercise the speed-alert flow. **15 km/h is a demo threshold, not the posted road limit.**
- The interface and exported records explicitly mark this calibration as `DEMO`. Displayed speeds are approximate and can change substantially if the assumed dimensions change.
- Real deployment needs measured ground dimensions or survey/camera calibration, the actual posted speed limit, and validation against a reliable reference speed. A demo configuration should not be described as accurate speed measurement.
- The signal in `tr2.mp4` is an overlay. Red-zone motion alerts and observed boundary crossings are exported as different event types; neither the overlay nor the manually calibrated line proves a real traffic offence.

## Commands

Default demonstration calibration, 50 km/h comparison limit:

```powershell
.\.venv\Scripts\python.exe -X utf8 traffic_violation_detection1.py
```

Lower threshold to illustrate speed alerts:

```powershell
.\.venv\Scripts\python.exe -X utf8 traffic_violation_detection1.py --speed-limit 15
```

Save a smooth recording:

```powershell
.\.venv\Scripts\python.exe -X utf8 traffic_violation_detection1.py --headless --save-video --speed-limit 15
```

Use `--no-speed` to disable speed estimation, or `--speed-config path.json` to supply another explicit calibration. The source frames, cached vehicle numbers, estimated speeds, configured limit, calibration status, and evidence paths are saved for inspection.
