"""
download_model.py
-----------------
Downloads the trained best.pt from Google Drive.

Usage:
    python scripts/download_model.py
"""

import os
import subprocess
import sys


MODEL_ID  = "1_EYUmnBE3a3PP0_f71VFBMESSsxv43m2"
SAVE_PATH = os.path.join("models", "weights", "best.pt")


def main():
    # Install gdown if not available
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    if os.path.exists(SAVE_PATH):
        size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
        print(f"best.pt already exists ({size_mb:.1f} MB)")
        print(f"  Path: {SAVE_PATH}")
        print("  Delete it and re-run to re-download.")
        return

    print(f"Downloading best.pt...")
    gdown.download(id=MODEL_ID, output=SAVE_PATH, quiet=False)

    size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
    print(f"\nDone! Saved to: {SAVE_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
