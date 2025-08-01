# 📧 Spam Email Detection using Machine Learning and NLP 

This project implements a **Spam Detection System** using multiple machine learning models. It processes and classifies SMS/email messages as **spam** or **ham (not spam)** using classical NLP techniques and supervised learning.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Preprocessing Steps](#-preprocessing-steps)
- [Models Used](#-models-used)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
- [Installation](#-installation)
- [Visualizations](#-visualizations)

---

## 🔍 Overview

The goal is to identify whether a given message is **spam** or **ham** based on its content using classical machine learning models. Various preprocessing techniques are used for data cleaning, and multiple algorithms are trained and evaluated for performance comparison.

---

## 📂 Dataset

- **Source**: `spam.csv`
- **Rows**: 5,572 messages
- **Columns**:  
  - `v1` → Label (spam or ham)  
  - `v2` → Message text

---

## 🔄 Preprocessing Steps

1. Dropping unused columns (`Unnamed`)
2. Renaming columns: `v1` → `Category`, `v2` → `Message`
3. Removing duplicates
4. Lowercasing text
5. Removing:
   - HTML tags
   - URLs
   - Punctuation
   - Special characters
   - Numbers
   - Non-alphanumeric characters
6. Chat slang normalization using a custom dictionary
7. Stopword removal using NLTK
8. Stemming using `PorterStemmer`
9. Vectorization using `CountVectorizer` and `TfidfVectorizer`

---

## 🤖 Models Used

| Model                      | Classifier                     |
|---------------------------|--------------------------------|
| Logistic Regression       | `LogisticRegression`           |
| Multinomial Naive Bayes   | `MultinomialNB`                |
| Gaussian Naive Bayes      | `GaussianNB`                   |
| Support Vector Machine    | `SVC`                          |
| Decision Tree             | `DecisionTreeClassifier`       |
| Random Forest             | `RandomForestClassifier`       |
| Gradient Boosting         | `GradientBoostingClassifier`   |
| XGBoost                   | `XGBClassifier`                |

---

## 📊 Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score (approximated)
- Confusion Matrix

---

## ✅ Results

| Model                 | Accuracy | Precision | F1 (approx) |
|----------------------|----------|-----------|-------------|
| MultinomialNB         | 0.9695   | 0.9712    | 0.9704      |
| XGBoost               | 0.9677   | 0.9682    | 0.9679      |
| Logistic Regression   | 0.9659   | 0.9672    | 0.9665      |
| Gradient Boosting     | 0.9623   | 0.9631    | 0.9627      |
| Decision Tree         | 0.9480   | 0.9481    | 0.9480      |
| Random Forest         | 0.9148   | 0.9224    | 0.9186      |
| SVM                   | 0.9094   | 0.9180    | 0.9136      |
| GaussianNB            | 0.8583   | 0.9133    | 0.8848      |

> 📌 **Best Performing Model:** `Multinomial Naive Bayes`

---

## 🛠 Installation

```bash
git clone https://github.com/Soumaditya777/Spam-Email-Classifier.git
cd spam-email-classifier
pip install -r requirements.txt
```


## 📈 Visualizations

WordClouds for spam and ham messages

Pie chart showing spam vs ham distribution

Model comparison plot (accuracy, precision)



