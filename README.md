# AI-Powered Consumer Review Analyzer

An AI-based Python project that analyzes online consumer reviews using Machine Learning and Natural Language Processing (NLP). It estimates whether a review resembles human-written or computer-generated examples and independently classifies its sentiment as Positive, Negative, or Neutral.

Developed as part of the **CBSE Class XII Artificial Intelligence (843) Team Project — Session 2026–27**.

## Project Overview

Online consumer reviews influence purchasing decisions, but distinguishing human-written reviews from computer-generated ones can be challenging. This project uses Machine Learning to identify linguistic patterns in review text and estimate its likely authorship category.

The application uses a TF-IDF vectorizer and Logistic Regression classifier for authorship assessment, along with NLTK's VADER sentiment analyzer for sentiment classification.

**Important:** Predictions are estimates based on learned text patterns. They do not prove who wrote a review or whether its claims are truthful.

## Features

- **Authorship Assessment:** Estimates whether a review resembles the Original or Computer Generated class.
- **Human-Written Probability:** Displays the model's estimated probability for the Original class.
- **Sentiment Analysis:** Classifies review sentiment as Positive, Negative, or Neutral.
- **Prediction Explanations:** Displays predefined statements associated with the predicted category.
- **Multiple Reviews:** Allows users to analyze reviews consecutively through the command-line interface.

## AI Techniques

### TF-IDF Vectorization
Converts review text into numerical features representing words and word combinations.

### Logistic Regression
A supervised Machine Learning algorithm trained to distinguish the Original and Computer Generated classes.

### VADER Sentiment Analysis
Uses a lexicon-based approach to classify review sentiment as Positive, Negative, or Neutral.

### Machine Learning Pipeline
Combines text vectorization and classification into a reusable pipeline saved with Joblib.

## Dataset

**Dataset:** [Fake Reviews Dataset — Kaggle](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)

The raw dataset contains 40,432 rows and four columns:

| Column | Description |
|---|---|
| `category` | Product category |
| `rating` | Review rating |
| `label` | Original or Computer Generated label |
| `text_` | Review text |

The labels are mapped as follows:

- `OR` → Original
- `CG` → Computer Generated

The dataset is cleaned and prepared before model training. The model uses the review text and corresponding labels.

### Dataset Attribution and License

**Dataset:** Fake Reviews Dataset  
**Creator/Uploader:** Mexwell  
**Source:** https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset  
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)  
**License URL:** https://creativecommons.org/licenses/by/4.0/

Attribution is provided to the original dataset source. Any modifications made to the dataset should be identified where applicable.

The dataset license is separate from the GNU GPL v3.0 license used for this project's source code.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Joblib

## Repository Structure

```text
TCG-code/
├── dataset/
│   ├── README.md
│   └── fake reviews dataset.csv
├── LICENSE
├── README.md
├── inspect_dataset.py
├── main.py
├── requirements.txt
├── review_analyser.py
└── train_model.py
```

**Note:** The trained model file `review_pipeline.pkl` is not included in this repository. Each user must generate their own model by running the training script.

## Installation and Usage

### Prerequisites

Install Python and Git on your computer.

### Step 1: Clone the Repository

Replace the example URL with your actual GitHub repository URL.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd TCG-code
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify the Dataset

Ensure the dataset CSV is present at this exact location:

```text
dataset/fake reviews dataset.csv
```

### Step 4: Train Your Own Model

Run the training script:

```bash
python train_model.py
```

This script preprocesses the dataset, trains and evaluates the Machine Learning pipeline, and saves the trained model as:

```text
models/review_pipeline.pkl
```

The `models` directory must be created by the training script if it does not already exist.

### Step 5: Run the Application

After training completes successfully, run:

```bash
python main.py
```

Enter a consumer review when prompted to view its authorship assessment, estimated human-written probability, sentiment, and explanatory statements.

**Important:** Run the training script before running the application on a fresh installation, because the trained model is not included in this repository.

## Model Evaluation

The recorded test accuracy was **89.09%** on a held-out test set created using an 80:20 stratified train-test split with `random_state=42`.

This result reflects evaluation on the selected dataset and split. Performance may differ on reviews from other platforms, product categories, or newer generative AI systems.

## Limitations

- Predictions are based on patterns learned from the training dataset.
- Human-written reviews may be classified as computer-generated, and vice versa.
- The probability score does not establish actual authorship.
- Sentiment analysis may struggle with sarcasm, context, and nuanced opinions.
- The system does not verify factual accuracy, reviewer identity, purchase history, or sponsorship.
- Prediction explanations are predefined statements rather than detailed feature-level explanations.

## Future Scope

- Train and evaluate the model using larger and more diverse datasets.
- Test performance on newer AI-generated reviews.
- Explore multilingual review analysis.
- Investigate promotional-content detection.
- Develop a graphical user interface or browser extension.
- Improve model evaluation and prediction explanations.

## Code License

The source code of this project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

Users may use, study, modify, and redistribute the covered code subject to the license terms. Distributed derivative works covered by the GPL must comply with its applicable requirements.

See the [`LICENSE`](LICENSE) file or read the full license at:

https://www.gnu.org/licenses/gpl-3.0.html

The code license does not automatically replace the dataset's license or determine the licensing status of every trained model artifact.

## Project Information

- **Project Title:** AI-Powered Consumer Review Analyzer
- **Subject:** Artificial Intelligence (843)
- **Class:** XII
- **Academic Session:** 2026–27
- **Project Type:** CBSE AI Team Project

---

*Developed for educational purposes as a demonstration of Machine Learning and Natural Language Processing applied to consumer review analysis.*
