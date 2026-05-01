"""
live_camera.py
--------------
Runs fire & smoke detection on your laptop camera in real time.

Usage:
    python scripts/live_camera.py --weights models/weights/best.pt
    python scripts/live_camera.py --weights models/weights/best.pt --conf 0.3
    python scripts/live_camera.py --weights models/weights/best.pt --camera 1  # use second camera

Controls:
    q — quit
    s — save current frame to screenshots/
"""

import argparse
import os
import time
import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Live fire & smoke detection")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (0 = default webcam)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    args = parser.parse_args()

    # Load model
    model = YOLO(args.weights)
    print(f"Model loaded: {args.weights}")
    print(f"Classes: {model.names}")

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {w}x{h}")
    print("Press 'q' to quit, 's' to save a screenshot\n")

    os.makedirs("screenshots", exist_ok=True)
    frame_count = 0
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        frame_count += 1

        # Run detection
        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        result = results[0]
        annotated = result.plot()

        # Calculate FPS
        now = time.time()
        fps = frame_count / (now - fps_time) if now != fps_time else 0

        # Draw FPS on frame
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw detection count
        n_fire = sum(1 for b in result.boxes if int(b.cls) == 1)
        n_smoke = sum(1 for b in result.boxes if int(b.cls) == 0)
        if n_fire > 0:
            cv2.putText(annotated, f"FIRE: {n_fire}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if n_smoke > 0:
            cv2.putText(annotated, f"SMOKE: {n_smoke}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 153, 51), 2)

        # Show window
        cv2.imshow("Fire & Smoke Detection", annotated)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = f"screenshots/frame_{frame_count}.jpg"
            cv2.imwrite(path, annotated)
            print(f"Saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()
