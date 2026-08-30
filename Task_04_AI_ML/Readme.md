# 📊 Customer Churn Prediction System

## 📌 Project Overview

This project is an end-to-end Machine Learning application developed as part of **Devixo Solutions AI/ML Internship – Task 04**.

The system predicts whether a bank customer is likely to **churn (leave the bank)** or **stay**, based on customer information such as credit score, age, balance, tenure, number of products, geography, gender, credit card status, active membership, and estimated salary.

The project covers the complete Machine Learning workflow, from data preprocessing and model development to hyperparameter tuning, model saving, and deployment through an interactive **Streamlit web application**.

---

## 🎯 Objective

The main objective of this project is to build a complete and practical Machine Learning application that:

- Processes a real-world customer dataset
- Handles missing values and duplicate records
- Detects and handles outliers
- Encodes categorical variables
- Performs feature engineering
- Scales numerical features
- Trains multiple Machine Learning models
- Compares model performance
- Applies cross-validation
- Performs hyperparameter tuning
- Saves the final trained model
- Provides an interactive prediction interface using Streamlit
- Deploys the application online

---

## 📂 Dataset

The project uses the **Churn Modelling Dataset** containing:

- **10,000 customer records**
- **14 original columns**

### Original Features

- RowNumber
- CustomerId
- Surname
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary
- Exited

The target variable is:

**Exited**

Where:

- `0` = Customer stays
- `1` = Customer churns

---

# 🧹 Data Processing

## 1. Dataset Loading

The dataset was loaded using Pandas and inspected to understand its structure, dimensions, and available features.

Dataset Shape:

```text
(10000, 14)
2. Missing Values

Missing values were checked across all columns.

Result:

No missing values found.

All 14 columns contained zero missing values.

3. Duplicate Records

Duplicate records were checked and the result was:

Duplicate Rows: 0

Therefore, no duplicate rows required removal.

📊 Outlier Detection and Handling

Outliers were detected using the Interquartile Range (IQR) method.

The following numerical features were analyzed:

CreditScore
Age
Tenure
Balance
NumOfProducts
EstimatedSalary
Detected Outliers
Feature	Outliers
CreditScore	15
Age	359
Tenure	0
Balance	0
NumOfProducts	60
EstimatedSalary	0

Outliers were handled using IQR Capping, where extreme values were capped at the calculated lower and upper IQR boundaries instead of removing customer records.

🔤 Categorical Encoding

Categorical variables were converted into numerical format using One-Hot Encoding.

Encoded features include:

Geography_Germany
Geography_Spain
Gender_Male

The original categorical columns were transformed into machine-learning compatible numerical features.

After encoding:

Encoded Dataset Shape: (10000, 12)
⚙️ Feature Engineering

A new feature was created:

BalanceSalaryRatio
BalanceSalaryRatio = Balance / EstimatedSalary

This feature represents the customer's account balance relative to their estimated salary and can provide additional information for predicting customer churn.

After feature engineering:

Dataset Shape: (10000, 13)
🎯 Feature and Target Separation

The target variable was separated from the input features.

Features Shape: (10000, 12)
Target Shape: (10000,)

Target:

Exited
📏 Feature Scaling

Numerical features were scaled using StandardScaler to ensure that features with different ranges could be used effectively by the Machine Learning models.

Scaled Feature Shape:

(10000, 12)
✂️ Train/Test Split

The dataset was divided into training and testing sets using an 80/20 split.

Training Samples: 8000
Testing Samples: 2000

Result:

X_train: (8000, 12)
X_test : (2000, 12)
y_train: (8000,)
y_test : (2000,)
🤖 Machine Learning Models

Four classification models were trained and compared:

Random Forest
Support Vector Machine (SVM)
Gradient Boosting
Decision Tree
📈 Model Evaluation

The models were evaluated using:

Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix
Model Performance
Model	Accuracy	Precision	Recall	F1 Score	ROC-AUC
Random Forest	86.50%	79.65%	45.21%	57.68%	85.13%
SVM	86.15%	84.95%	38.82%	53.29%	82.31%
Gradient Boosting	87.25%	80.40%	49.39%	61.19%	86.80%
Decision Tree	79.75%	50.23%	53.56%	51.84%	70.00%
Best Initial Model

Based on the evaluation results, Gradient Boosting achieved the strongest overall performance.

Initial Accuracy:

87.25%

ROC-AUC:

86.80%

F1 Score:

61.19%
🔄 5-Fold Cross-Validation

To evaluate model stability and generalization, 5-Fold Cross-Validation was performed.

Model	Mean CV Accuracy	Standard Deviation
Random Forest	86.21%	0.80%
SVM	85.54%	0.65%
Gradient Boosting	86.25%	0.86%
Decision Tree	79.01%	0.58%

Gradient Boosting achieved the highest mean cross-validation accuracy:

86.25%
🎛️ Hyperparameter Tuning

GridSearchCV was used to optimize the Gradient Boosting model.

Best Parameters
learning_rate = 0.1
max_depth = 4
n_estimators = 100

Best Cross-Validation Accuracy:

86.42%

The tuned Gradient Boosting model was selected as the final model.

🏆 Final Tuned Model Evaluation

The tuned Gradient Boosting model was evaluated on the test dataset.

Metric	Score
Accuracy	86.90%
Precision	78.88%
Recall	48.65%
F1 Score	60.18%
ROC-AUC	87.01%
Confusion Matrix
[[1540   53]
 [ 209  198]]

The final model achieved an ROC-AUC of 87.01%, showing good ability to distinguish between customers who are likely to churn and customers who are likely to stay.

💾 Model Saving

The final trained Gradient Boosting model was saved using Joblib.

Model location:

models/best_model.joblib

This saved model is used by the Streamlit application to generate predictions without retraining the model every time.

🌐 Streamlit Application

A professional interactive web interface was developed using Streamlit.

Users can enter:

Credit Score
Age
Tenure
Balance
Number of Products
Geography
Gender
Credit Card Status
Active Membership
Estimated Salary

After entering the information, the user can click the prediction button to receive:

Customer churn prediction
Churn probability
Customer input summary

Example prediction:

Customer is likely to CHURN
Churn Probability: 99.24%
🚀 Deployment

The Streamlit application has been deployed online using Streamlit Community Cloud.

Live Application

https://devixosolution-aug6qd7gqwdzgzaevqt9fu.streamlit.app/

The deployed application allows users to interact with the trained Machine Learning model directly through a web browser.

🛠️ Technologies Used
Programming Language
Python
Libraries
Pandas
NumPy
Scikit-learn
Matplotlib
Joblib
Streamlit
Machine Learning
Random Forest
Support Vector Machine
Gradient Boosting
Decision Tree
GridSearchCV
5-Fold Cross-Validation
StandardScaler
One-Hot Encoding
Tools
Git
GitHub
VS Code
Streamlit Community Cloud
📁 Project Structure
Task_04_AI_ML/
│
├── DataSet/
│   └── Churn_Modelling.csv
│
├── models/
│   └── best_model.joblib
│
├── app.py
├── streamlit_app.py
├── requirements.txt
└── README.md
▶️ How to Run Locally
1. Clone the Repository
git clone https://github.com/shamii-101/DEVIOX_SOLUTION.git
2. Navigate to Task 04
cd DEVIOX_SOLUTION/Task_04_AI_ML
3. Install Dependencies
pip install -r requirements.txt
4. Run the Streamlit Application
python -m streamlit run streamlit_app.py

The application will open in the browser at:

http://localhost:8501
🎯 Key Outcomes

Through this project, the following practical Machine Learning skills were implemented:

Real-world dataset processing
Data quality checking
Missing value analysis
Duplicate detection
IQR-based outlier detection
Outlier capping
Categorical encoding
Feature engineering
Feature scaling
Train/test splitting
Multiple classification algorithms
Model performance comparison
Confusion matrix analysis
ROC-AUC evaluation
5-Fold Cross-Validation
Hyperparameter tuning using GridSearchCV
Model serialization using Joblib
Interactive Streamlit application development
Online ML application deployment
👨‍💻 Internship Task

Organization: Devixo Solutions
Program: AI/ML Internship
Task: Task 04 – End-to-End ML Application with Deployment

This project demonstrates the complete transition from a Machine Learning notebook/workflow to a working, user-facing Machine Learning application.


