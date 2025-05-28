# Language-Detection-model

📌 Overview
This project aims to build a system that can automatically detect the language of a given piece of text using Machine Learning and Deep Learning techniques. It was implemented as part of my Capstone Project (CP2) during my Data Science course at Imarticus Learning.

🧠 Objective
To develop and compare different models that can accurately classify the input text into one of several languages. The project uses real-world multilingual text data and explores both traditional ML techniques and Deep Learning.
📂 Dataset
Source: [Multilingual Text Dataset]

Size: ~22 languages, with 1000 rows per language

Columns: Text, Language Label

Preprocessing: Cleaning, Label Encoding, Tokenization (for DL)

🔍 Problem Statement
In a world full of global communication, understanding the language of a text is crucial. From chatbots to translation tools and spam filtering, automatic language detection helps improve customer experience, localization, and data filtering.
The problem is to build a multi-class classification model that can predict the correct language of any given sentence from the dataset.


🛠️ Technologies & Libraries
Python
Pandas, Numpy, Matplotlib, Seaborn
Scikit-learn (Naive Bayes, Logistic Regression)
TensorFlow & Keras (LSTM)
Gradio (Deployment UI)
Pickle (Model Serialization)

📈 Models Implemented
1. Multinomial Naive Bayes
Best Performing Model
Accuracy: ~96%
Simple, fast, and interpretable
Used CountVectorizer for text feature extraction

2. Logistic Regression
Compared with Naive Bayes
Slightly lower performance
Useful for understanding feature importance

3. LSTM (Deep Learning)
Tokenized input + padded sequences
Lower accuracy than Naive Bayes on small data
More scalable with bigger datasets.

✅ Evaluation Metrics
Confusion Matrix
Classification Report (Precision, Recall, F1-Score)
Accuracy Score
Model Comparison using visualization

🌍 Real-world Applications
This project has potential use in:
Chatbots and Customer Support Systems
Social Media Monitoring
Spam Detection
Automated Language Routing
Content Moderation Platforms

🚀 Deployment
Used Gradio to create a simple web interface:
Input: User types a sentence
Output: Detected language shown in real-time
Backend: Naive Bayes model served using Pickle

🧠 What’s New / Innovative?
Compared ML and DL models on the same task

Deployed real-time predictions via Gradio

Demonstrated practical model usage with user input

Used confusion matrix and heatmaps to visualize performance

Worked with Pickle to save/load models for deployment

🔮 Future Scope
Improve DL accuracy with a larger dataset
Integrate translation and speech-to-text modules
Deploy as a Flask or Streamlit app for production
Train with more diverse languages and dialects
Add Explainable AI (XAI) insights.

