# 📧 Spam Mail Prediction using Machine Learning

This project focuses on building a **Machine Learning model** that classifies email messages as **Spam** or **Not Spam (Ham)** based on the message content.  
It demonstrates the use of **Natural Language Processing (NLP)** and **text classification techniques** for email filtering.

---

## 📘 Project Overview

With the massive growth of digital communication, spam emails have become a serious issue.  
This project uses machine learning and text analysis to automatically detect whether an incoming email is **spam** or **legitimate** based on its text features.

---

## 🧠 Objective

To develop an **email spam detection system** that:
- Learns from historical labeled email data.
- Predicts whether a new email is **Spam** or **Ham**.

---

## 🧩 Dataset Details

- **Dataset Name:** Spam Mail Dataset (SMS Spam Collection or similar)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Number of Instances:** ~5,500 messages  
- **Columns:**
  - `Category` → Spam or Ham  
  - `Message` → Text content of the email/sms  

---

## ⚙️ Technologies Used

- Python 🐍  
- Jupyter Notebook  
- NumPy  
- Pandas  
- Scikit-learn  
- NLTK (Natural Language Toolkit)  
- Matplotlib / Seaborn  

---

## 🧮 Steps Performed

1. **Data Loading and Exploration**
   - Loaded the dataset using Pandas.
   - Checked data balance and basic statistics.

2. **Text Preprocessing**
   - Converted all text to lowercase.
   - Removed punctuation, stopwords, and special symbols.
   - Tokenized text and applied stemming/lemmatization.

3. **Feature Extraction**
   - Used **TF-IDF Vectorizer** or **CountVectorizer** to convert text into numerical features.

4. **Model Training**
   - Applied **Naive Bayes Classifier (MultinomialNB)** — best suited for text classification.
   - Split data into **training** and **testing** sets.

5. **Model Evaluation**
   - Evaluated with metrics:
     - Accuracy
     - Confusion Matrix
     - Precision, Recall, F1-Score
   - Verified model generalization on test data.

6. **Prediction System**
   - Implemented a simple function to classify a user-input email as spam or not spam.

---

## 📊 Results

| Metric | Train Accuracy | Test Accuracy |
|:-------:|:---------------:|:--------------:|
| Naive Bayes | ~98% | ~97% |

*(Results may slightly vary depending on preprocessing and random split)*

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Varugowdatp/spam_mail_prediction.git
