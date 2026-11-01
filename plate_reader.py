"""Local plate-specific OCR. No registration numbers are supplied to the model."""

from pathlib import Path

import cv2
import numpy as np

from anpr_core import normalize_plate

ROOT = Path(__file__).resolve().parent


class PlateReader:
    def __init__(self):
        import onnxruntime as ort
        from fast_plate_ocr import LicensePlateRecognizer

        model = ROOT / "models/cct_s_v2_global.onnx"
        config = ROOT / "models/cct_s_v2_global_plate_config.yaml"
        if not model.is_file() or not config.is_file():
            raise FileNotFoundError("OCR weights missing. Run: python download_plate_model.py")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.reader = LicensePlateRecognizer(onnx_model_path=model, plate_config_path=config,
                                              device="cpu", sess_options=options)

    def read_plate(self, crop, pattern):
        if crop.size == 0 or min(crop.shape[:2]) < 6:
            return None
        # The global CCT model expects RGB uint8 images; resizing is handled by it.
        image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        prediction = self.reader.run(image, return_confidence=True)[0]
        value = normalize_plate(prediction.plate, pattern)
        if value is None:
            return None
        probabilities = np.asarray(prediction.char_probs[:len(value)], dtype=float)
        confidence = float(np.exp(np.log(np.clip(probabilities, 1e-8, 1)).mean()))
        return (value, confidence) if confidence >= 0.5 else None
