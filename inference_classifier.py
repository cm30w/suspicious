"""Run real-time ASL alphabet recognition from the webcam."""

import pickle

import cv2
import mediapipe as mp

MODEL_PATH = "model.p"
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
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Try changing the device index in VideoCapture(0)."
        )

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]

                    h, w, _ = frame.shape
                    x1 = int(min(x_coords) * w) - 20
                    y1 = int(min(y_coords) * h) - 20
                    x2 = int(max(x_coords) * w) + 20
                    y2 = int(max(y_coords) * h) + 20

                    features = extract_features(hand_landmarks)
                    prediction = int(model.predict([features])[0])
                    letter = chr(ord("A") + prediction)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(
                        frame,
                        letter,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA,
                    )

            cv2.imshow("ASL Alphabet Classifier", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
