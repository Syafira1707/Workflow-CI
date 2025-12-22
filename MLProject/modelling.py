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
# Training model
# =========================
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# =========================
# Evaluasi
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# =========================
# Logging ke MLflow
# =========================
mlflow.log_metric("accuracy", accuracy)

# SIMPAN MODEL
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model"
)

print("Training CI via MLflow Project BERHASIL.")
