# Customer Churn Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting whether a customer is likely to **churn (leave)** or **stay** with a company.

A Customer Churn Prediction system can help businesses identify customers who may leave and take preventive actions to improve customer retention.

The project implements multiple machine learning classification models and compares their performance using different evaluation metrics. It also includes cross-validation, hyperparameter tuning, feature importance analysis, confusion matrices, and model saving using Joblib.

---

## 🎯 Objectives

- Prepare and clean the customer churn dataset.
- Perform feature engineering and feature selection.
- Encode categorical variables.
- Scale feature values.
- Train multiple machine learning classification models.
- Evaluate models using Accuracy, Precision, Recall, and F1 Score.
- Generate confusion matrices.
- Apply 5-Fold Cross-Validation.
- Perform hyperparameter tuning using GridSearchCV.
- Analyze feature importance.
- Compare the performance of different models.
- Select and save the best-performing model.
- Load the saved model and verify its performance.

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

The dataset satisfies the minimum requirement of 3,000 records.

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

A new feature called `BalanceSalaryRatio` was created to represent the relationship between the customer's balance and estimated salary.

```python
X["BalanceSalaryRatio"] = X["Balance"] / (X["EstimatedSalary"] + 1)
Feature Selection

Irrelevant identifier and unnecessary columns were removed before model training.

Categorical Encoding

Categorical variables such as Geography and Gender were converted into numerical features using One-Hot Encoding.

X = pd.get_dummies(
    X,
    columns=["Geography", "Gender"],
    drop_first=True
)

The final encoded dataset contained:

10,000 × 12

features.

Feature Scaling

StandardScaler was used to standardize the feature values.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Train/Test Split

The dataset was divided into:

Training data: 8,000 records
Testing data: 2,000 records

An 80/20 split was used with stratification.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
🤖 Machine Learning Models

Four machine learning classification models were trained:

Random Forest
Support Vector Machine (SVM)
Gradient Boosting
Decision Tree

After evaluating the initial models, Gradient Boosting was selected for hyperparameter tuning because it achieved the highest mean cross-validation accuracy among the initial models.

📈 Model Evaluation

The models were evaluated using:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Final Test Results
Model	Accuracy	Precision	Recall	F1 Score
Random Forest	86.65%	80.17%	45.70%	58.22%
SVM	85.95%	83.87%	38.33%	52.61%
Gradient Boosting	86.95%	78.52%	49.39%	60.63%
Decision Tree	79.35%	49.31%	52.33%	50.77%
Tuned Gradient Boosting	87.10%	82.25%	46.68%	59.56%
Result Analysis

The Tuned Gradient Boosting model achieved the highest test accuracy of 87.10%.

The original Gradient Boosting model achieved the highest F1 Score of 60.63%.

The SVM model achieved the highest Precision of 83.87% among the initial models.

For the final model selection, the Tuned Gradient Boosting Classifier was selected primarily because it achieved the highest test accuracy while maintaining strong precision.

🔲 Confusion Matrix

Confusion matrices were generated for all four initial machine learning models to analyze correct and incorrect predictions.

A confusion matrix contains:

True Negatives (TN)
False Positives (FP)
False Negatives (FN)
True Positives (TP)
Gradient Boosting Confusion Matrix
[[1538   55]
 [ 206  201]]

This means:

1538 customers were correctly predicted as staying.
55 customers were incorrectly predicted as churned.
206 churned customers were missed.
201 churned customers were correctly identified.
🔄 Cross-Validation

5-Fold Cross-Validation was applied to the training data to evaluate model consistency across different subsets of the training dataset.

Model	Mean CV Accuracy	Standard Deviation
Random Forest	86.18%	0.67%
SVM	85.49%	0.62%
Gradient Boosting	86.20%	0.88%
Decision Tree	78.81%	0.40%

Gradient Boosting achieved the highest mean cross-validation accuracy among the initial models.

🎛️ Hyperparameter Tuning

GridSearchCV was used to optimize the Gradient Boosting model.

The following hyperparameters were tested:

n_estimators
learning_rate
max_depth

A total of 27 parameter combinations were evaluated using 5-Fold Cross-Validation.

Best Parameters
learning_rate = 0.05
max_depth = 4
n_estimators = 100
Best Cross-Validation Accuracy
86.40%

After tuning, the model achieved:

87.10% test accuracy
🔍 Feature Importance

Feature importance was extracted from the tuned Gradient Boosting model to identify which features contributed most to the model's predictions.

Feature Importance Results
Rank	Feature	Importance
1	Age	38.60%
2	NumOfProducts	28.56%
3	IsActiveMember	12.27%
4	Geography_Germany	6.56%
5	Balance	4.85%
6	BalanceSalaryRatio	3.46%
7	EstimatedSalary	2.02%
8	CreditScore	1.98%
9	Gender_Male	1.23%
10	Tenure	0.33%
11	Geography_Spain	0.08%
12	HasCrCard	0.06%

The results show that Age, Number of Products, and Active Member status were the most influential features for the trained Gradient Boosting model.

🏆 Best Model

The final selected model is:

Tuned Gradient Boosting Classifier

The model was selected primarily based on its highest test accuracy of 87.10%.

Final Test Performance
Accuracy  : 87.10%
Precision : 82.25%
Recall    : 46.68%
F1 Score  : 59.56%
Cross-Validation Performance
Best CV Accuracy : 86.40%
💾 Model Saving and Loading

The trained model was saved using Joblib.

import joblib

joblib.dump(best_model, "models/best_model.joblib")

The saved model was then loaded again and tested successfully.

loaded_model = joblib.load("models/best_model.joblib")

loaded_accuracy = loaded_model.score(X_test, y_test)

print(f"Loaded Model Test Accuracy: {loaded_accuracy:.4f}")
Loaded Model Test Accuracy
87.10%

This confirms that the trained model can be loaded and reused without retraining.

📁 Project Structure
Task_03_AI_ML/
│
├── dataset/
│   └── churn.csv
│
├── models/
│   └── best_model.joblib
│
├── screenshots/
│   ├── data_preparation.png
│   ├── model_results.png
│   ├── confusion_matrix.png
│   ├── cross_validation.png
│   ├── hyperparameter_tuning.png
│   ├── feature_importance.png
│   └── final_comparison.png
│
├── main.py
├── requirements.txt
└── README.md
▶️ How to Run
1. Open the Project

Open the Task_03_AI_ML folder in VS Code.

2. Install Required Libraries

Run:

pip install -r requirements.txt
3. Run the Machine Learning Pipeline

Run:

python main.py

The script performs:

Dataset loading
Data preprocessing
Feature engineering
Feature selection
Categorical encoding
Feature scaling
Train/Test splitting
Model training
Model evaluation
Confusion matrix generation
Cross-validation
Hyperparameter tuning
Feature importance analysis
Model comparison
Model saving
Saved model loading and verification
⚠️ Limitations
The model is trained on a single customer churn dataset.
Model performance may change when applied to different datasets.
Recall for churned customers is lower than overall accuracy.
The model should be validated using real-world business data before production deployment.
Additional techniques may be required to improve the detection of churned customers.
🚀 Future Improvements

Possible future improvements include:

Testing additional machine learning models such as XGBoost.
Applying class imbalance handling techniques.
Improving churn recall.
Adding probability-based churn prediction.
Building an interactive Streamlit web interface.
Deploying the trained model as an API.
Monitoring model performance after deployment.
Adding real-time customer churn prediction.
🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Joblib
📌 Project Status

Task 03 - Completed ✅

The complete customer churn prediction machine learning pipeline was implemented, optimized, evaluated, and documented.

The final Tuned Gradient Boosting model achieved:

87.10% test accuracy

The trained model was successfully saved as a .joblib file and loaded again to verify that it can be reused without retraining.
