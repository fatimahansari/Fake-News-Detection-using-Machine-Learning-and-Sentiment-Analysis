# Fake News Detection using Machine Learning and Sentiment Analysis

## Project Overview
This project focuses on detecting fake news using machine learning algorithms and sentiment analysis. The system classifies news statements into categories such as True, False, or Partially True based on their textual content. It combines natural language processing (NLP) techniques with classical ML models to analyze and predict the credibility of news statements.

## Features
- Text preprocessing and cleaning for effective NLP analysis
- Sentiment analysis using TextBlob to enrich features
- TF-IDF vectorization for converting text into numerical features
- Handling class imbalance using SMOTE
- Machine learning classifiers implemented: Naive Bayes, Decision Tree, KNN, Logistic Regression
- K-Means clustering for unsupervised analysis
- Performance evaluation with accuracy, classification reports, and confusion matrices
- Visualization of sentiment distribution, label distribution, and model performance
- Model and vectorizer serialization using `joblib` for future use

## Tech Stack
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, TextBlob, Imbalanced-learn, Matplotlib, Seaborn, Joblib, SciPy
- **Techniques:** Classification, Clustering, TF-IDF, SMOTE Oversampling
- **Visualization:** Matplotlib and Seaborn

## Dataset
- Train, Test, and Validation datasets are loaded from TSV files
- Columns: `ID, Label, Statement, Subject, Speaker, Job Title, State, Party, BTC, FC, HTC, MTC, POFC, Context`
- For modeling, only `Statement` and `Label` are used

