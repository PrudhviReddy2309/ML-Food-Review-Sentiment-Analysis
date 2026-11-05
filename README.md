# Food Review Sentiment Analysis Using Machine Learning

## 1. Project Overview

Food Review Sentiment Analysis is a Natural Language Processing (NLP) and Machine Learning project designed to identify customer opinions from restaurant reviews.

The system automatically classifies customer reviews into two sentiment categories:

* Positive
* Negative

Multinomial Naive Bayes is the selected classification algorithm.

## 2. Project Objectives

* Analyse customer food reviews.
* Identify positive and negative opinions.
* Convert textual reviews into numerical features.
* Compare different classification algorithms.
* Predict the sentiment of new customer reviews.

## 3. Dataset Description

The dataset contains 9,000 food review records and 9 columns.

### Selected Features

| Column              | Description               |
| ------------------- | ------------------------- |
| Review_ID           | Unique review identifier  |
| Review              | Customer review text      |
| Liked               | Sentiment target (0 or 1) |
| Cuisine_Type        | Cuisine category          |
| Review_Date         | Date of review            |
| Rating              | Overall rating            |
| Price_Range         | Price category            |
| Service_Rating      | Service score             |
| Food_Quality_Rating | Food quality score        |

**Target Variable: Liked**

* 0 = Negative
* 1 = Positive

**Dataset Note:** This is a synthetic educational dataset and does not represent verified real-world restaurant feedback.

## 4. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Natural Language Processing (NLP)
* TF-IDF Vectorizer

## 5. Machine Learning Algorithms

The following algorithms were compared:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear SVM
4. Random Forest
5. K-Nearest Neighbors (KNN)

**Selected Algorithm: Multinomial Naive Bayes**

Multinomial Naive Bayes is suitable for text classification because it uses probabilistic learning based on feature values.

## 6. Methodology and Workflow

### Step 1: Dataset Collection

Load the food review dataset containing review text and sentiment labels.

### Step 2: Data Preprocessing

* Remove unnecessary characters.
* Clean the review text.
* Handle missing values.
* Prepare sentiment labels.

### Step 3: Text Feature Extraction

Use TF-IDF (Term Frequency-Inverse Document Frequency) to convert textual reviews into numerical feature vectors.

Unigram and bigram features are used.

* Unigram: Individual words.
* Bigram: Combinations of two consecutive words.

### Step 4: Train-Test Splitting

The dataset is divided into:

* Training records: 7,200
* Testing records: 1,800

### Step 5: Model Training

Train and compare five algorithms:

* Multinomial Naive Bayes
* Logistic Regression
* Linear SVM
* Random Forest
* KNN

### Step 6: Model Evaluation

Evaluate the models using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Step 7: Sentiment Prediction

The selected Multinomial Naive Bayes model predicts whether a new review expresses positive or negative sentiment.

## 7. Model Performance

Reported Multinomial Naive Bayes results:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 88.33% |
| Precision | 88.65% |
| Recall    | 87.86% |
| F1-score  | 88.26% |

## 8. Confusion Matrix

Multinomial Naive Bayes confusion matrix:

|                 | Predicted Negative | Predicted Positive |
| --------------- | -----------------: | -----------------: |
| Actual Negative |                801 |                101 |
| Actual Positive |                109 |                789 |

### Interpretation

* True Negative (TN): 801
* False Positive (FP): 101
* False Negative (FN): 109
* True Positive (TP): 789

The diagonal values represent correct predictions, while the off-diagonal values represent incorrect predictions.

## 9. How Sentiment Prediction Works

Example 1:

**Input Review:**

"The food was delicious and the service was excellent."

Workflow:

1. Accept customer review.
2. Clean the review text.
3. Convert the text into numerical features using TF-IDF.
4. Pass the features to Multinomial Naive Bayes.
5. Calculate the probability of each sentiment class.
6. Predict the class with the higher probability.

**Expected Output: Positive**

Example 2:

**Input Review:**

"The food was cold and the service was very poor."

**Expected Output: Negative**

## 10. Execution Workflow

1. Load the dataset.
2. Preprocess customer reviews.
3. Extract TF-IDF features.
4. Split the dataset into training and testing sets.
5. Train five classification algorithms.
6. Compare model performance.
7. Select Multinomial Naive Bayes.
8. Generate the confusion matrix and performance plots.
9. Enter a new review.
10. Display the predicted sentiment.

## 11. Conclusion

The Food Review Sentiment Analysis project demonstrates how NLP and Machine Learning can automatically identify customer opinions from textual reviews.

Multinomial Naive Bayes was selected with 88.33% accuracy and 88.26% F1-score on the reported test set.

The system can help analyse customer feedback and understand overall sentiment.

## 12. Future Enhancements

* Integrate real restaurant review datasets.
* Develop a Streamlit web application.
* Support multiple languages.
* Improve sentiment detection for sarcasm and complex expressions.
* Generate customer satisfaction reports.
