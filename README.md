# 📝 Review Classifier Using Similarity Scores

A unique approach to classify reviews (mental-health-based) using **text similarity** instead of traditional supervised classifiers.

## 💡 Project Overview

This project implements a review sentiment classifier using **cosine similarity** of input reviews against prototypical "positive" and "negative" review sets. It leverages NLP-based preprocessing, vectorization, and similarity computation to assign a sentiment score to unseen reviews.

---

## 🧰 Tech Stack

- Python
- Pandas, NumPy
- scikit-learn
- CountVectorizer
- Cosine Similarity

---

## 🔍 Approach

1. **Preprocessing**
   - Lowercasing, punctuation removal, stopword filtering, stemming
2. **Vectorization**
   - CountVectorizer to convert text into feature vectors
3. **Similarity Score Computation**
   - Calculate cosine similarity of a new review with known review samples
4. **Classification Logic**
   - Assign label based on which class the new review is more similar to

---



## 📦 How to Run

```bash
git clone https://github.com/PragyaPathak147/ReviewClassifier.git
cd ReviewClassifier
pip install -r requirements.txt
