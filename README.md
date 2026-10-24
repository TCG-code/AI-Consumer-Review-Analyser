# AI-Powered Consumer Review Analyzer

An AI-based Python project that analyzes online consumer reviews using Machine Learning and Natural Language Processing (NLP). The system estimates whether a review resembles human-written or computer-generated examples and independently identifies its sentiment as Positive, Negative, or Neutral.

This project was developed as part of the **CBSE Class XII Artificial Intelligence (843) Team Project, Session 2026–27**.

## 📌 Project Overview

Online consumer reviews influence purchasing decisions, but it can be difficult to distinguish naturally written reviews from computer-generated ones. This project explores how Machine Learning can identify linguistic patterns in review text to estimate its likely authorship category.

The application uses a trained Machine Learning model for authorship assessment and NLTK's VADER sentiment analyzer for sentiment classification.

**Note:** The system provides probability-based predictions, not definitive proof of who wrote a review or whether its claims are truthful.

## ✨ Features

- **Authorship Assessment:** Estimates whether a review resembles human-written or computer-generated samples from the training dataset.
- **Human-Written Probability:** Displays the model's estimated probability for the Original class.
- **Sentiment Analysis:** Classifies the review as Positive, Negative, or Neutral.
- **Prediction Explanation:** Displays predefined explanatory statements associated with the predicted category.
- **Multiple Reviews:** Allows users to analyze reviews consecutively through the command-line interface.

## 🧠 AI Techniques Used

### 1. TF-IDF Vectorization
Converts review text into numerical features that represent the importance of words and word combinations.

### 2. Logistic Regression
A supervised Machine Learning algorithm trained to classify reviews into the Original and Computer Generated categories.

### 3. VADER Sentiment Analysis
Analyzes the sentiment expressed in review text and classifies it as Positive, Negative, or Neutral.

### 4. Machine Learning Pipeline
Combines text vectorization and classification into a single pipeline saved for reuse.

## 📊 Dataset

**Dataset:** [Fake Reviews Dataset — Kaggle](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)

The dataset contains labelled consumer reviews used to train and evaluate the authorship classification model.

The raw dataset contains the following columns:

| Column | Description |
|---|---|
| `category` | Product category |
| `rating` | Review rating |
| `label` | Original or Computer Generated label |
| `text_` | Review text |

The model uses the review text and labels for training. The labels are mapped as follows:

- `OR` → Original
- `CG` → Computer Generated

The dataset contains 40,432 rows before preprocessing. Cleaning and duplicate removal produce the data used for model training.

### Dataset License and Attribution

The dataset is provided under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license, according to the license information provided for the dataset.

**Attribution:**

Mexwell, *Fake Reviews Dataset*, Kaggle.  
Source: https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset  
License: https://creativecommons.org/licenses/by/4.0/

The dataset is credited to its original source. Any modifications made to the dataset should be identified when applicable. Refer to the dataset's source page for its attribution and licensing details.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Joblib

## 📁 Project Structure

```text
AI-Consumer-Review-Analyzer/
├── dataset/
│   └── fake reviews dataset.csv
├── models/
│   └── review_pipeline.pkl
├── inspect_dataset.py
├── train_model.py
├── review_analyser.py
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
```

## ⚙️ Installation and Usage

### Prerequisites

Install Python and Git on your computer.

### Step 1: Clone the Repository

```bash
git clone https://github.com/TCG-code/AI-Consumer-Review-Analyser
cd AI-Consumer-Review-Analyzer
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository's actual URL.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Prepare the Files

Ensure that the dataset is located at:

```text
dataset/fake reviews dataset.csv
```

Ensure that the trained model is located at:

```text
models/review_pipeline.pkl
```

If the trained model is not available, run the training script after placing the dataset in the correct location:

```bash
python train_model.py
```

### Step 4: Run the Application

```bash
python main.py
```

Enter a consumer review when prompted to receive its authorship assessment, estimated human-written probability, sentiment, and explanatory statements.

## 📈 Model Evaluation

The model was evaluated using a stratified 80:20 train-test split with `random_state=42`.

The recorded test accuracy was **89.09%** on the held-out test set.

This result reflects performance on the selected dataset and split. It does not guarantee the same performance on reviews from other platforms, product categories, or newer generative AI systems.

## ⚠️ Limitations

- The model learns patterns from its training dataset and may not generalize to all types of reviews.
- Human-written reviews may be classified as computer-generated, and computer-generated reviews may be classified as human-written.
- The probability score is not proof of actual authorship.
- Sentiment classification may not correctly interpret sarcasm, context, or nuanced opinions.
- The system does not verify factual accuracy, reviewer identity, purchase history, or sponsorship.
- The explanatory statements are predefined and are not detailed, feature-level explanations of the model's internal decisions.

## 🔮 Future Scope

Potential improvements include:

- Training with larger and more diverse datasets.
- Evaluating performance on newer AI-generated reviews.
- Supporting multiple languages.
- Exploring additional review-quality and promotional-content indicators.
- Developing a graphical user interface or browser extension.
- Improving model evaluation and prediction explanations.

## 📜 Code License

The source code of this project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

Under this license, users may use, study, modify, and redistribute the covered code subject to the license terms. Distributed derivative works covered by the GPL must comply with its applicable requirements, including copyleft provisions.

See the [`LICENSE`](LICENSE) file or read the full license at:

https://www.gnu.org/licenses/gpl-3.0.html

**Important:** The project's code license and the dataset's CC BY 4.0 license are separate. The GPL license for the code does not automatically replace the dataset's license or determine the licensing status of every generated model artifact.

### Dataset Attribution and License

**Dataset Name:** Fake Reviews Dataset  
**Creator/Uploader:** Mexwell  
**Source:** https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset  
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)  
**License Link:** https://creativecommons.org/licenses/by/4.0/

The dataset is used for educational and research purposes in this project. Attribution is provided to the original dataset source. Any modifications made to the dataset will be identified where applicable.

The dataset's license is separate from the GNU GPL v3.0 license used for this project's source code.


## 👥 Project Information

- **Project Title:** AI-Powered Consumer Review Analyzer
- **Subject:** Artificial Intelligence (843)
- **Class:** XII
- **Academic Session:** 2026–27
- **Project Type:** CBSE AI Team Project

---

*Developed for educational purposes as a demonstration of Machine Learning and Natural Language Processing applied to consumer review analysis.*
