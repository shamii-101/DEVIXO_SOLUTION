# Customer Churn Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting whether a customer is likely to **churn (leave)** or **stay** with a company.

A Customer Churn Prediction system can help businesses identify customers who may leave and take preventive actions to improve customer retention.

The project implements multiple machine learning classification models and compares their performance using different evaluation metrics.

---

## 🎯 Objectives

- Prepare and clean the customer churn dataset.
- Perform feature engineering and feature selection.
- Encode categorical variables.
- Scale numerical features.
- Train multiple machine learning classification models.
- Evaluate models using Accuracy, Precision, Recall, and F1 Score.
- Apply 5-Fold Cross-Validation.
- Perform hyperparameter tuning using GridSearchCV.
- Analyze feature importance.
- Select and save the best-performing model.
- Verify that the saved model can be loaded and used for prediction.

---

## 📊 Dataset

The project uses a customer churn dataset containing:

- **10,000 customer records**
- **14 original columns**

The dataset contains customer information such as:

- Credit Score
- Age
- Tenure
- Balance
- Number of Products
- Credit Card status
- Active Member status
- Estimated Salary
- Geography
- Gender
- Customer Churn status

The dataset satisfies the minimum 3,000-record requirement.

---

## ⚙️ Data Preparation

The following preprocessing steps were performed:

1. Dataset loading using Pandas
2. Missing value checking
3. Duplicate checking
4. Feature engineering
5. Feature selection
6. Categorical encoding
7. Feature scaling
8. Train/Test splitting

### Feature Engineering

A new feature called:

```text
BalanceSalaryRatio