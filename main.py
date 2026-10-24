from review_analyser import analyse_review


def display_heading():
    """Display the application heading."""

    print("\n" + "=" * 68)
    print("AI-POWERED CONSUMER REVIEW AUTHORSHIP ANALYSER".center(68))
    print("=" * 68)

    print(
        "\nThis system estimates whether a consumer review resembles "
        "human-written or computer-generated text."
    )

    print(
        "It also analyses the review's sentiment as Positive, "
        "Negative, or Neutral."
    )


def get_user_review():
    """Receive a consumer review from the user."""

    print("\nEnter or paste a consumer review below.")
    print("Press Enter after typing the review.\n")

    return input("Review: ").strip()


def display_analysis(analysis):
    """Display the four final outputs returned by review_analyser.py."""

    print("\n" + "=" * 68)
    print("CONSUMER REVIEW AUTHORSHIP ANALYSIS".center(68))
    print("=" * 68)

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

    print("\n" + "-" * 68)

    print(
        "Note: This result is a probability-based prediction derived from "
        "linguistic patterns."
    )

    print(
        "It does not verify whether the review is truthful, sponsored, "
        "or written by a verified buyer."
    )

    print("=" * 68)


def ask_to_continue():
    """Ask whether the user wants to analyse another review."""

    while True:
        choice = input(
            "\nWould you like to analyse another review? (yes/no): "
        ).strip().lower()

        if choice in {"yes", "y"}:
            return True

        if choice in {"no", "n"}:
            return False

        print("Please enter 'yes' or 'no'.")


def main():
    """Run the complete user-facing application."""

    display_heading()

    while True:
        review_text = get_user_review()

        if not review_text:
            print("\nError: The review cannot be empty.")
            continue

        try:
            analysis = analyse_review(review_text)
            display_analysis(analysis)

        except (ValueError, TypeError, FileNotFoundError) as error:
            print(f"\nError: {error}")

        except Exception as error:
            print("\nAn unexpected error occurred.")
            print(f"Details: {error}")

        if not ask_to_continue():
            print("\nThank you for using the Consumer Review Authorship Analyser.")
            print("Application closed successfully.\n")
            break


if __name__ == "__main__":
    main()