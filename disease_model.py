import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle

# Load expanded dataset
df = pd.read_csv("dataset_expanded.csv")

# Features & Target
X = df[["age", "blood_pressure", "glucose_level", "heart_rate"]]
y = df["diagnosis"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline with scaling + logistic regression with higher max_iter
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

# Perform 5-fold cross-validation on training data
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.3f}")

# Fit model on the full training set
model.fit(X_train, y_train)

# Save trained pipeline (scaler + model)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
print("Accuracy on training set:", model.score(X_train, y_train))
print("Accuracy on test set:", model.score(X_test, y_test))
(venv) mohannarayanapuram@Mohans-MacBook-Air healthcare_project % python disease_model.py

Cross-validation accuracy scores: [0.83333333 0.83333333 0.83333333 0.82352941 0.88235294]
Mean CV accuracy: 0.841
✅ Model trained and saved as model.pkl
Accuracy on training set: 0.8977272727272727
Accuracy on test set: 0.7272727272727273
(venv) mohannarayanapuram@Mohans-MacBook-Air healthcare_project % 