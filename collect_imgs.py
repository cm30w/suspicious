"""Collect raw webcam images for each ASL alphabet class (A–Z)."""

import os
import time

import cv2

DATA_DIR = "./data"
NUM_CLASSES = 26
IMAGES_PER_CLASS = 200
CAPTURE_DELAY_SEC = 0.05


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Try changing the device index in VideoCapture(0)."
        )

    for class_id in range(NUM_CLASSES):
        letter = chr(ord("A") + class_id)
        class_dir = os.path.join(DATA_DIR, str(class_id))
        os.makedirs(class_dir, exist_ok=True)

        waiting = True
        while waiting:
            ret, frame = cap.read()
            if not ret:
                continue

            display = frame.copy()
            cv2.putText(
                display,
                f"Class {class_id} ({letter}): press Q to capture {IMAGES_PER_CLASS} images",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Collect ASL Data", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                waiting = False

        print(f"Collecting {IMAGES_PER_CLASS} images for {letter}...")
        for img_idx in range(IMAGES_PER_CLASS):
            ret, frame = cap.read()
            if not ret:
                continue

            path = os.path.join(class_dir, f"{img_idx}.jpg")
            cv2.imwrite(path, frame)

            cv2.putText(
                frame,
                f"{letter}: {img_idx + 1}/{IMAGES_PER_CLASS}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Collect ASL Data", frame)
            cv2.waitKey(1)
            time.sleep(CAPTURE_DELAY_SEC)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Images saved under {DATA_DIR}/")


if __name__ == "__main__":
    main()
