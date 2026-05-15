"""Extract normalized hand landmarks from collected images."""

import os
import pickle

import cv2
import mediapipe as mp

DATA_DIR = "./data"
OUTPUT_PATH = "data.pickle"
MIN_DETECTION_CONFIDENCE = 0.3


def extract_features(landmarks) -> list[float]:
    x_coords = [lm.x for lm in landmarks.landmark]
    y_coords = [lm.y for lm in landmarks.landmark]
    min_x = min(x_coords)
    min_y = min(y_coords)

    features: list[float] = []
    for x, y in zip(x_coords, y_coords):
        features.append(x - min_x)
        features.append(y - min_y)
    return features


def main() -> None:
    mp_hands = mp.solutions.hands

    data: list[list[float]] = []
    labels: list[int] = []

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    ) as hands:
        for class_id in sorted(os.listdir(DATA_DIR), key=int):
            class_path = os.path.join(DATA_DIR, class_id)
            if not os.path.isdir(class_path):
                continue

            label = int(class_id)
            for image_name in os.listdir(class_path):
                if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                image_path = os.path.join(class_path, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    continue

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                if not results.multi_hand_landmarks:
                    continue

                features = extract_features(results.multi_hand_landmarks[0])
                data.append(features)
                labels.append(label)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)

    print(f"Saved {len(data)} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
