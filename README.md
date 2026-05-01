# PCCC_yolo

Fire and smoke detection using YOLOv8, designed for CCTV cameras in mini-apartments.

PCCC = Phong Chay Chua Chay (Fire Prevention & Fighting)

## Dataset

- **fire-smoke-new** — 115,845 train / 16,100 val images
- Classes: `smoke`, `fire`

## Project Structure

```
PCCC_yolo/
├── notebooks/
│   ├── 01_download_dataset.ipynb    # Load & inspect dataset (Colab)
│   ├── 02_train.ipynb               # Train YOLOv8s (Colab)
│   └── 03_inference_demo.ipynb      # Run on demo videos (Colab)
├── scripts/
│   ├── download_model.py            # Download best.pt from Drive
│   └── live_camera.py               # Real-time detection on laptop camera
├── models/weights/                  # Model weights (gitignored)
├── data/                            # Dataset files (gitignored)
└── data.yaml                        # YOLO dataset config
```

## Quick Start

### Download model

```bash
pip install ultralytics gdown
python scripts/download_model.py
```

### Live camera detection

```bash
python scripts/live_camera.py --weights models/weights/best.pt
```

Press `q` to quit, `s` to save a screenshot.

### Train on Colab

Open `notebooks/02_train.ipynb` in Google Colab and run all cells.

## Results

Trained on Google Colab with T4 GPU using YOLOv8s for 50 epochs.
