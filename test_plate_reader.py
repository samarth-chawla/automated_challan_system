"""Regression on the real crop EasyOCR previously misread as WB645."""

from pathlib import Path
import unittest

import cv2

from anpr_core import PLATE_PATTERN
from plate_reader import PlateReader

ROOT = Path(__file__).resolve().parent


@unittest.skipUnless((ROOT / "models/cct_s_v2_global.onnx").is_file(), "Download OCR assets first")
class ActualPlateTests(unittest.TestCase):
    def test_visible_bmw_plate_is_read_from_pixels(self):
        image = cv2.imread(str(ROOT / "tests/fixtures/bmw_plate.png"))
        result = PlateReader().read_plate(image, PLATE_PATTERN)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "YB6433")
        self.assertGreater(result[1], 0.7)


if __name__ == "__main__":
    unittest.main()
