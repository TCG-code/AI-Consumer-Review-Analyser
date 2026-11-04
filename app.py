import streamlit as st
from review_analyser import analyse_review

# Page configuration
st.set_page_config(
    page_title="AI Consumer Review Analyzer",
    page_icon="🔍",
    layout="centered"
)

# Header
st.title("🔍 AI-Powered Consumer Review Analyzer")
st.caption("Machine Learning + Natural Language Processing")

st.markdown(
    """
    Analyze consumer reviews using Artificial Intelligence.
    Estimate review authorship patterns and identify sentiment.
    """
)

st.divider()

# Review input
st.subheader("Enter a Consumer Review")

review_text = st.text_area(
    "Paste your review below:",
    placeholder="Example: The product has excellent sound quality, "
                "but the battery life could be better...",
    height=160
)

# Analysis
if st.button("🔍 Analyse Review", type="primary", use_container_width=True):

    if not review_text.strip():
        st.warning("Please enter a review before analysing.")

    else:
        try:
            with st.spinner("Analysing your review..."):
                result = analyse_review(review_text)

            st.divider()
            st.subheader("Analysis Results")

            probability = result["human_written_probability"]

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Human-Written Probability",
                    f"{probability:.2f}%"
                )

            with col2:
                st.metric(
                    "Review Sentiment",
                    result["sentiment"]
                )

            st.subheader("Authorship Assessment")
            st.info(result["classification"])

            st.progress(probability / 100)

            st.subheader("Prediction Explanation")

            for number, reason in enumerate(result["reasons"], start=1):
                st.write(f"**{number}.** {reason}")

            st.divider()

            st.caption(
                "Disclaimer: This is a probability-based prediction "
                "derived from linguistic patterns. It does not verify "
                "actual authorship, factual accuracy, sponsorship, "
                "or verified purchase status."
            )

        except FileNotFoundError:
            st.error(
                "Trained model not found. Please run "
                "'python train_model.py' before starting the application."
            )

        except Exception as error:
            st.error(f"An error occurred: {error}")

# Footer
st.divider()
st.caption("CBSE Class XII | Artificial Intelligence (843) | 2026–27")
