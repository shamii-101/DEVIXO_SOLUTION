import pandas as pd
import numpy as np

df = pd.read_csv(r"D:\devoix solution\Task_03_AI_ML\dataset\Churn_Modelling.csv")


print(df.shape)
print(df.info())
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
print("\nFeature Engineering:")
print(df[["Balance", "EstimatedSalary", "BalanceSalaryRatio"]].head())

df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
print("\nColumns after Feature Selection:")
print(df.columns)

X = df.drop(columns=["Exited"])
y = df["Exited"]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(y.name)

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

X = pd.get_dummies(X, columns=["Geography", "Gender"], drop_first=True)

print("\nEncoded Features:")
print(X.columns)
print("\nEncoded Shape:", X.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Shape:", X_scaled.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Features:", X_train.shape)
print("Testing Features:", X_test.shape)
print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Initialize Models

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

svm = SVC(
    kernel="rbf",
    random_state=42
)

gradient_boosting = GradientBoostingClassifier(
    random_state=42
)

decision_tree = DecisionTreeClassifier(
    random_state=42
)

# Train Models

random_forest.fit(X_train, y_train)
svm.fit(X_train, y_train)
gradient_boosting.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)

print("\nAll 4 Models Trained Successfully!")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predictions

rf_pred = random_forest.predict(X_test)
svm_pred = svm.predict(X_test)
gb_pred = gradient_boosting.predict(X_test)
dt_pred = decision_tree.predict(X_test)

# Evaluation Function

def evaluate_model(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{name}")
    print("-" * 30)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


# Evaluate all models

evaluate_model("Random Forest", y_test, rf_pred)
evaluate_model("SVM", y_test, svm_pred)
evaluate_model("Gradient Boosting", y_test, gb_pred)
evaluate_model("Decision Tree", y_test, dt_pred)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

models_predictions = {
    "Random Forest": rf_pred,
    "SVM": svm_pred,
    "Gradient Boosting": gb_pred,
    "Decision Tree": dt_pred
}

for name, predictions in models_predictions.items():

    cm = confusion_matrix(y_test, predictions)

    print(f"\n{name} Confusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Stayed", "Churned"]
    )

    disp.plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    from sklearn.model_selection import cross_val_score

models = {
    "Random Forest": random_forest,
    "SVM": svm,
    "Gradient Boosting": gradient_boosting,
    "Decision Tree": decision_tree
}

print("\nCross-Validation Results")
print("=" * 40)

cv_results = {}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    cv_results[name] = scores

    print(f"\n{name}")
    print(f"Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f}")
    print(f"Standard Deviation: {scores.std():.4f}")

    from sklearn.model_selection import GridSearchCV

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

# Get the best tuned model

best_model = grid_search.best_estimator_

# Make predictions on test data

tuned_pred = best_model.predict(X_test)

# Evaluate tuned model

print("\nTuned Gradient Boosting Model")
print("-" * 35)

print(f"Accuracy : {accuracy_score(y_test, tuned_pred):.4f}")
print(f"Precision: {precision_score(y_test, tuned_pred):.4f}")
print(f"Recall   : {recall_score(y_test, tuned_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, tuned_pred):.4f}")

import pandas as pd
import matplotlib.pyplot as plt

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Gradient Boosting - Feature Importance")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "SVM",
        "Gradient Boosting",
        "Decision Tree",
        "Tuned Gradient Boosting"
    ],
    "Accuracy": [
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, svm_pred),
        accuracy_score(y_test, gb_pred),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, tuned_pred)
    ],
    "Precision": [
        precision_score(y_test, rf_pred),
        precision_score(y_test, svm_pred),
        precision_score(y_test, gb_pred),
        precision_score(y_test, dt_pred),
        precision_score(y_test, tuned_pred)
    ],
    "Recall": [
        recall_score(y_test, rf_pred),
        recall_score(y_test, svm_pred),
        recall_score(y_test, gb_pred),
        recall_score(y_test, dt_pred),
        recall_score(y_test, tuned_pred)
    ],
    "F1 Score": [
        f1_score(y_test, rf_pred),
        f1_score(y_test, svm_pred),
        f1_score(y_test, gb_pred),
        f1_score(y_test, dt_pred),
        f1_score(y_test, tuned_pred)
    ]
})

print("\nFinal Model Comparison:")
print(comparison.round(4).to_string(index=False))


# Accuracy Comparison Graph

plt.figure(figsize=(10, 6))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

plt.xticks(rotation=20)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

import joblib
joblib.dump(best_model, "best_model.joblib")
print("\nBest model saved successfully as: best_model.joblib")

# Load the saved model

loaded_model = joblib.load("best_model.joblib")

# Make predictions using the loaded model

loaded_pred = loaded_model.predict(X_test)

# Check accuracy

loaded_accuracy = accuracy_score(y_test, loaded_pred)

print("\nLoaded Model Test Accuracy:")
print(f"{loaded_accuracy:.4f}")

print("\nSaved model loaded and tested successfully!")