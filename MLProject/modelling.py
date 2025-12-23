import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow
import mlflow.sklearn


def train_model(data_path: str):
    # ===============================
    # 1. LOAD DATA PREPROCESSING
    # ===============================
    df = pd.read_csv(data_path)

    target_col = "Survived"
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols]
    y = df[target_col]

    # ===============================
    # 2. TRAIN TEST SPLIT
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ===============================
    # 3. MLFLOW SETUP
    # ===============================
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Titanic_RandomForest_Manual")

    with mlflow.start_run(run_name="rf_manual_run"):
        # ===============================
        # 4. TRAIN MODEL
        # ===============================
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)

        # ===============================
        # 5. EVALUATION
        # ===============================
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        print("Accuracy:", acc)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # ===============================
        # 6. LOG METRICS & PARAMS (WAJIB)
        # ===============================
        mlflow.log_metric("accuracy", acc)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("test_size", 0.2)

        # ===============================
        # 7. SAVE MODEL LOCAL
        # ===============================
        output_dir = Path("artifacts")
        output_dir.mkdir(exist_ok=True)

        model_path = output_dir / "random_forest_titanic.pkl"
        joblib.dump(model, model_path)

        # ===============================
        # 8. LOG ARTIFACTS (INI KUNCI)
        # ===============================
        mlflow.log_artifact(model_path)

        print(f"\nModel disimpan di: {model_path}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "titanic_preprocessed.csv"

    train_model(DATA_PATH)
