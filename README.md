# Blinkit Review Classifier

## Overview
This repository contains an end-to-end Natural Language Processing (NLP) application designed to classify Blinkit customer reviews. As part of a comprehensive data science portfolio, this project demonstrates a complete machine learning pipeline—from text processing to deployment. It utilizes TF-IDF vectorization and a Support Vector Machine (SVM) classifier to accurately analyze and categorize unstructured text feedback.

An interactive web interface, built with Streamlit, is included to allow users (and recruiters) to input sample reviews and test the model's inference in real-time.

## Project Structure
- `blinkit_review_analyzer_v2.ipynb`: The core Jupyter Notebook documenting data exploration, text preprocessing, feature engineering (TF-IDF), and the training of the SVM model.
- `portfolio_app/app.py`: The Streamlit web application serving as the frontend for model interaction.
- `portfolio_app/svm_classifier.pkl`: The serialized Support Vector Machine model, saved after training.
- `portfolio_app/tfidf_vectorizer.pkl`: The fitted TF-IDF vectorizer used to transform raw input text into numerical features.
- `requirements.txt`: Python package dependencies required to run the application.

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd blinkit-review-classifier
   ```

2. **Install the required dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   Launch the web interface locally using:
   ```bash
   streamlit run portfolio_app/app.py
   ```

## Technical Stack
- **Language:** Python
- **Machine Learning & NLP:** Scikit-learn (SVM, TF-IDF)
- **Deployment & UI:** Streamlit
- **Data Manipulation:** Pandas, NumPy

## Author
**Kavya Agrawal**  
*Data Scientist & Machine Learning Engineer*
