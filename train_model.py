from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "fake reviews dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "review_pipeline.pkl"


def load_and_prepare_data():
    """Load the dataset and prepare review text and labels."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATASET_PATH}\n"
            "Check the CSV filename and folder location."
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = {"text_", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain these columns: {required_columns}"
        )

    # Keep only the required columns
    df = df[["text_", "label"]].copy()

    # Remove empty rows and duplicates
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["text_"], inplace=True)

    # Convert labels into meaningful names
    label_mapping = {
        "OR": "Original",
        "CG": "Computer Generated"
    }

    df["label"] = df["label"].map(label_mapping)

    # Remove any unexpected labels
    df.dropna(subset=["label"], inplace=True)

    X = df["text_"]
    y = df["label"]

    return X, y, df


def train_model():
    """Train, test and save the review authenticity model."""

    print("Loading dataset...")

    X, y, df = load_and_prepare_data()

    print(f"Total usable reviews: {len(df)}")
    print("\nClass distribution:")
    print(y.value_counts())

    # 80% training data and 20% testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining reviews: {len(X_train)}")
    print(f"Testing reviews: {len(X_test)}")

    # Complete NLP + ML pipeline
    pipeline = Pipeline([
        (
            "vectorizer",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=30000,
                ngram_range=(1, 2),
                min_df=2
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    print("Testing model...")
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Create models folder if it does not exist
    MODEL_DIR.mkdir(exist_ok=True)

    # Save vectorizer and model together
    joblib.dump(pipeline, MODEL_PATH)

    print(f"\nModel saved successfully at:\n{MODEL_PATH}")


if __name__ == "__main__":
    train_model()