import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn


def train_model(data_path: str):
    # ===============================
    # 1. LOAD DATA
    # ===============================
    df = pd.read_csv(data_path)

    target_col = "Survived"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ===============================
    # 2. TRAIN TEST SPLIT
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ===============================
    # 3. MLFLOW SETUP
    # ===============================
    mlflow.set_experiment("Titanic_RandomForest_Autolog")

    # Autolog sebagai logging utama
    mlflow.sklearn.autolog(log_models=False)

    # ===============================
    # 4. TRAINING
    # ===============================
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ===============================
    # 5. LOG MODEL
    # ===============================
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "titanic_preprocessed.csv"

    train_model(DATA_PATH)
