# Real-Time ASL Alphabet Classifier

A computer vision system that recognizes American Sign Language (ASL) alphabet gestures in real time using hand landmark detection and machine learning.

## Overview

This project recognizes ASL alphabet signs (A–Z) from a webcam feed. [MediaPipe](https://developers.google.com/mediapipe) tracks 21 hand landmarks per frame; a Random Forest classifier predicts the letter from normalized landmark coordinates. It is useful for ASL practice, accessibility prototypes, and learning end-to-end ML pipelines.

## Features

- **Real-time recognition** — Interactive webcam inference
- **26-class classifier** — Full ASL alphabet (A–Z)
- **Position invariant** — Landmarks normalized to the hand bounding box
- **Visual feedback** — Landmarks, bounding box, and predicted letter on screen
- **Custom data pipeline** — Collect, preprocess, train, and deploy your own model

## Tech Stack

| Component | Role |
|-----------|------|
| MediaPipe | 21 hand keypoints per hand |
| OpenCV | Capture, display, and image I/O |
| scikit-learn | Random Forest classifier |
| NumPy | Numerical arrays |
| pickle | Feature store and model serialization |

## Architecture

```
Webcam → MediaPipe Hands → Normalize 42 features → Random Forest → On-screen letter
```

**Features:** 21 landmarks × (x, y), each coordinate shifted by `min(x)` and `min(y)` for position invariance.

**Model:** Random Forest — 200 trees, max depth 20, `min_samples_split=5`, 80/20 train/test split.

## Getting Started

### Prerequisites

- Python 3.9+
- Webcam

```bash
pip install -r requirements.txt
```

### Pipeline

#### 1. Collect training images

```bash
python collect_imgs.py
```

- For each letter, press **Q** to start capturing 200 frames
- Hold the ASL sign steady; vary position slightly across sessions for robustness
- Images are saved under `data/0/` … `data/25/` (A–Z)

#### 2. Build the feature dataset

```bash
python create_dataset.py
```

- Runs MediaPipe on every image
- Writes `data.pickle` with features and labels

#### 3. Train the classifier

```bash
python train_classifier.py
```

- Trains on 80% of samples, reports holdout accuracy
- Saves `model.p`

#### 4. Run live inference

```bash
python inference_classifier.py
```

- Press **Q** or **ESC** to quit

## Project Structure

```
.
├── collect_imgs.py          # Webcam data collection
├── create_dataset.py        # Landmark extraction
├── train_classifier.py      # Train Random Forest
├── inference_classifier.py  # Live recognition
├── requirements.txt
├── data/                    # Raw images (created by you)
├── data.pickle              # Features (generated)
└── model.p                  # Trained model (generated)
```

## Tips

- Use even lighting and a plain background when collecting data
- Vary hand distance and angle slightly per class
- If the camera fails to open, try `cv2.VideoCapture(1)` in the scripts
- Lower `min_detection_confidence` in `create_dataset.py` / `inference_classifier.py` if hands are missed

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| Camera not opening | Close other apps using the webcam; change capture index to `1` |
| Low accuracy | Recollect with more pose/lighting variety; ensure full hand is visible |
| Hand not detected | Improve lighting; lower `MIN_DETECTION_CONFIDENCE` |

## Future Ideas

- Confidence scores and temporal smoothing
- Words, numbers, and two-handed signs
- Web UI or CNN-based model
- Dataset augmentation

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [MediaPipe](https://developers.google.com/mediapipe) for hand tracking
- ASL gesture references: [Start ASL alphabet](https://www.startasl.com/american-sign-language-alphabet/)
