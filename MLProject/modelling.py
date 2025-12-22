import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================
# Load dataset
# =========================
df = pd.read_csv("titanic_preprocessed.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# Train model
# =========================
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# =========================
# Evaluate
# =========================
accuracy = accuracy_score(y_test, model.predict(X_test))


# =========================
# MLflow logging
# (RUN SUDAH ADA DARI MLflow Project)
# =========================
mlflow.log_param("model", "LogisticRegression")
mlflow.log_param("max_iter", 200)

mlflow.log_metric("accuracy", accuracy)

mlflow.sklearn.log_model(model, "model")

print("CI training selesai dengan sukses.")
