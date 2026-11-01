"""Download the plate detector and plate-specific OCR assets with SHA256 checks."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

MODEL_URL = ("https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/"
             "83c98fbe7412fe8b3950adb5637cfd08b0f04809/best.pt")
MODEL_SHA256 = "2d95861825bb4184404344c9cf809f40fd31dba785fe54e8ba5b9a3583789822"


def download(target, url, expected_sha256):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha256:
        print(f"Verified existing model: {target}")
        return
    temporary = target.with_suffix(".download")
    print(f"Downloading {target.name}...")
    try:
        with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError(f"Model checksum mismatch: {target.name}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Verified model saved to {target}")


def main():
    root = Path(__file__).resolve().parent
    download(root / "license_plate.pt", MODEL_URL, MODEL_SHA256)
    base = "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/"
    for filename, checksum in [
        ("cct_s_v2_global.onnx", "384bbbd2cea3ef54761d3df70822ef3a349ee1a112aeafddbe0e3ba06bc6e47b"),
        ("cct_s_v2_global_plate_config.yaml", "0335c74a305173bb6f393efed0fde03cadeaa0b649ed8e19f431016d8232d0a6"),
    ]:
        download(root / "models" / filename, base + filename, checksum)


if __name__ == "__main__":
    main()
