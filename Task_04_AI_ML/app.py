import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    r"D:\devoix solution\Task_04_AI_ML\DataSet\Churn_Modelling.csv"
)

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ==========================================
# 2. OUTLIER DETECTION USING IQR
# ==========================================

numerical_columns = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary"
]

print("\nOutlier Detection using IQR")
print("=" * 50)

outlier_counts = {}

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    outlier_counts[column] = len(outliers)

    print(f"\n{column}")
    print(f"Lower Bound : {lower_bound:.2f}")
    print(f"Upper Bound : {upper_bound:.2f}")
    print(f"Outliers    : {len(outliers)}")


# ==========================================
# 3. OUTLIER SUMMARY
# ==========================================

print("\nOutlier Summary")
print("=" * 50)

for column, count in outlier_counts.items():
    print(f"{column}: {count}")

# ==========================================
# 3. OUTLIER HANDLING USING IQR CAPPING
# ==========================================

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column] = df[column].clip(
        lower_bound,
        upper_bound
    )

print("\nOutliers handled using IQR Capping.")
# ==========================================
# CATEGORICAL ENCODING
# ==========================================

print("\nCategorical Encoding")
print("=" * 50)

# Remove unnecessary columns
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

# One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    drop_first=True
)

print("\nEncoded Dataset:")
print(df.head())

print("\nEncoded Dataset Shape:")
print(df.shape)

print("\nEncoded Columns:")
print(df.columns)
# ==========================================
# FEATURE ENGINEERING
# ==========================================

print("\nFeature Engineering")
print("=" * 50)

# Create Balance to Salary Ratio
df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

print("\nNew Feature Created: BalanceSalaryRatio")

print("\nUpdated Dataset:")
print(df.head())

print("\nUpdated Dataset Shape:")
print(df.shape)
# ==========================================
# FEATURE / TARGET SEPARATION
# ==========================================

print("\nFeature and Target Separation")
print("=" * 50)

X = df.drop("Exited", axis=1)
y = df["Exited"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)


# ==========================================
# FEATURE SCALING
# ==========================================

from sklearn.preprocessing import StandardScaler

print("\nFeature Scaling")
print("=" * 50)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Scaled Features Shape:", X_scaled.shape)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain/Test Split")
print("=" * 50)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)
# ==========================================
# MACHINE LEARNING MODELS
# ==========================================

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

print("\nTraining Machine Learning Models")
print("=" * 50)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": SVC(
        probability=True,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    )
}

trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    trained_models[name] = model

    print(f"{name} training completed.")

print("\nAll 4 models trained successfully!")
# ==========================================
# MODEL EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

print("\nModel Evaluation")
print("=" * 50)

results = []

for name, model in trained_models.items():

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    print(f"\n{name}")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)


# ==========================================
# FINAL MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

print("\nFinal Model Comparison")
print("=" * 70)
print(results_df.to_string(index=False))
# ==========================================
# 5-FOLD CROSS-VALIDATION
# ==========================================

from sklearn.model_selection import cross_val_score

print("\n5-Fold Cross-Validation")
print("=" * 50)

cv_results = []

for name, model in trained_models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    mean_score = scores.mean()
    std_score = scores.std()

    cv_results.append({
        "Model": name,
        "Mean CV Accuracy": mean_score,
        "Standard Deviation": std_score
    })

    print(f"\n{name}")
    print(f"Fold Scores: {scores}")
    print(f"Mean Accuracy: {mean_score:.4f}")
    print(f"Standard Deviation: {std_score:.4f}")


cv_df = pd.DataFrame(cv_results)

print("\nCross-Validation Summary")
print("=" * 70)
print(cv_df.to_string(index=False))
# ==========================================
# HYPERPARAMETER TUNING
# ==========================================

from sklearn.model_selection import GridSearchCV

print("\nHyperparameter Tuning - Gradient Boosting")
print("=" * 50)

param_grid = {
    "n_estimators": [100, 150, 200],
    "learning_rate": [0.05, 0.1, 0.15],
    "max_depth": [2, 3, 4]
}

grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(f"{grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

print("\nBest Tuned Model:")
print(best_model)
# ==========================================
# TUNED MODEL FINAL EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

print("\nTuned Gradient Boosting - Final Evaluation")
print("=" * 50)

# Predictions
tuned_predictions = best_model.predict(X_test)
tuned_probabilities = best_model.predict_proba(X_test)[:, 1]

# Metrics
tuned_accuracy = accuracy_score(y_test, tuned_predictions)
tuned_precision = precision_score(y_test, tuned_predictions)
tuned_recall = recall_score(y_test, tuned_predictions)
tuned_f1 = f1_score(y_test, tuned_predictions)
tuned_roc_auc = roc_auc_score(y_test, tuned_probabilities)

print(f"Accuracy : {tuned_accuracy:.4f}")
print(f"Precision: {tuned_precision:.4f}")
print(f"Recall   : {tuned_recall:.4f}")
print(f"F1 Score : {tuned_f1:.4f}")
print(f"ROC-AUC  : {tuned_roc_auc:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tuned_predictions))
# ==========================================
# SAVE FINAL MODEL
# ==========================================

import os
import joblib

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/best_model.joblib")

print("\nModel Saving")
print("=" * 50)
print("Final model saved successfully!")
print("Location: models/best_model.joblib")