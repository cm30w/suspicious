"""Train a Random Forest on hand landmark features."""

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATASET_PATH = "data.pickle"
MODEL_PATH = "model.p"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main() -> None:
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    x_train, x_test, y_train, y_test = train_test_split(
        dataset["data"],
        dataset["labels"],
        test_size=TEST_SIZE,
        shuffle=True,
        stratify=dataset["labels"],
        random_state=RANDOM_STATE,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    print(f"{accuracy * 100}% of samples were classified correctly")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model}, f)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
