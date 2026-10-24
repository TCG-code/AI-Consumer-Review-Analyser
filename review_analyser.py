from pathlib import Path

import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "review_pipeline.pkl"


# ---------------------------------------------------------
# CLASSIFICATION THRESHOLDS
# ---------------------------------------------------------

GENUINE_THRESHOLD = 80.0
SUSPICIOUS_THRESHOLD = 50.0


# ---------------------------------------------------------
# PREDEFINED REASON STATEMENTS
# Four statements are displayed for each classification.
# ---------------------------------------------------------

REASON_STATEMENTS = {
    "Likely Human-Written": [
        "The review resembles original human-written samples in the training dataset.",
        "The model detected linguistic patterns associated with the Original class.",
        "The prediction shows high probability of human-written authorship.",
        "The result is based on textual patterns learned during model training."
    ],

    "Uncertain Authorship": [
        "The model produced moderate probability of human-written authorship.",
        "The text contains patterns associated with both trained classes.",
        "The prediction is inconclusive based on the available dataset.",
        "The review falls within the model's uncertainty range."
    ],

    "Likely Computer-Generated": [
        "The review resembles computer-generated samples in the training dataset.",
        "The model detected patterns associated with the Computer Generated class.",
        "The prediction shows low probability of human-written authorship.",
        "The result is based on patterns learned from labelled training reviews."
    ]
}

# ---------------------------------------------------------
# LOAD TRAINED MACHINE LEARNING MODEL
# ---------------------------------------------------------

def load_review_model():
    """
    Load the previously trained Scikit-learn pipeline.

    The pipeline contains:
    1. TF-IDF text vectorizer
    2. Logistic Regression classifier
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model was not found at:\n{MODEL_PATH}\n\n"
            "Run train_model.py before using the review analyser."
        )

    return joblib.load(MODEL_PATH)


# Load the model once when this file is imported.
review_pipeline = load_review_model()


# ---------------------------------------------------------
# LOAD NLTK VADER SENTIMENT ANALYSER
# ---------------------------------------------------------

def load_sentiment_analyser():
    """
    Load the NLTK VADER sentiment analyser.

    The VADER lexicon is downloaded only if it is not already
    available on the computer.
    """

    try:
        return SentimentIntensityAnalyzer()

    except LookupError:
        print("Downloading the NLTK VADER sentiment resource...")
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


sentiment_analyser = load_sentiment_analyser()


# ---------------------------------------------------------
# AUTHENTICITY SCORE
# ---------------------------------------------------------

def calculate_human_written_probability(review_text):
    """
    Calculate the probability that the review belongs to
    the 'Original' class.

    Returns a percentage between 0 and 100.
    """

    probabilities = review_pipeline.predict_proba([review_text])[0]

    classifier = review_pipeline.named_steps["classifier"]
    class_names = list(classifier.classes_)

    if "Original" not in class_names:
        raise ValueError(
            "The trained model does not contain the expected "
            "'Original' class."
        )

    original_class_index = class_names.index("Original")
    original_probability = probabilities[original_class_index]

    authenticity_score = original_probability * 100

    return round(authenticity_score, 2)


# ---------------------------------------------------------
# REVIEW CLASSIFICATION
# ---------------------------------------------------------

def classify_review(human_written_probability):
    if human_written_probability >= GENUINE_THRESHOLD:
        return "Likely Human-Written"

    if human_written_probability >= SUSPICIOUS_THRESHOLD:
        return "Uncertain Authorship"

    return "Likely Computer-Generated"

# ---------------------------------------------------------
# SENTIMENT ANALYSIS
# ---------------------------------------------------------

def analyse_sentiment(review_text):
    """
    Analyse the review sentiment using NLTK VADER.

    Returns:
    Positive, Negative or Neutral
    """

    sentiment_scores = sentiment_analyser.polarity_scores(review_text)
    compound_score = sentiment_scores["compound"]

    if compound_score >= 0.05:
        sentiment = "Positive"

    elif compound_score <= -0.05:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return sentiment


# ---------------------------------------------------------
# REASON STATEMENTS
# ---------------------------------------------------------

def get_reason_statements(classification):
    """
    Return the four predefined reasoning statements belonging
    to the predicted classification.
    """

    return REASON_STATEMENTS[classification].copy()


# ---------------------------------------------------------
# COMPLETE REVIEW ANALYSIS
# ---------------------------------------------------------

def analyse_review(review_text):
    """
    Perform the complete review analysis.

    Final outputs:
    1. Authenticity Score
    2. Review Classification
    3. Sentiment Analysis
    4. AI Reasoning Statements
    """

    if not isinstance(review_text, str):
        raise TypeError("The review must be provided as text.")

    cleaned_review = review_text.strip()

    if not cleaned_review:
        raise ValueError("The review cannot be empty.")

    human_written_probability = calculate_human_written_probability(cleaned_review)

    classification = classify_review(human_written_probability)

    sentiment = analyse_sentiment(cleaned_review)

    reasons = get_reason_statements(classification)

    result = {
    "human_written_probability": human_written_probability,
    "classification": classification,
    "sentiment": sentiment,
    "reasons": reasons
    }

    return result


# ---------------------------------------------------------
# TEMPORARY TESTING SECTION
# This section runs only when review_analyser.py is executed
# directly. It will not run when imported into main.py.
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("AI-POWERED CONSUMER REVIEW AUTHORSHIP ANALYSER")
    print("=" * 60)

    user_review = input("\nEnter a consumer review:\n\n")

    try:
        analysis = analyse_review(user_review)

        print("\n" + "=" * 60)
        print("CONSUMER REVIEW AUTHORSHIP ANALYSIS")
        print("=" * 60)

        print(
            f"\nHuman-Written Probability: "
            f"{analysis['human_written_probability']:.2f}%"
        )

        print(
            f"\nAuthorship Assessment: "
            f"{analysis['classification']}"
        )

        print(
            f"\nReview Sentiment: "
            f"{analysis['sentiment']}"
        )

        print("\nPrediction Explanation:")

        for number, reason in enumerate(analysis["reasons"], start=1):
            print(f"{number}. {reason}")

        print("\n" + "=" * 60)

    except (ValueError, TypeError, FileNotFoundError) as error:
        print(f"\nError: {error}")