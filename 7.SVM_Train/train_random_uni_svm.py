import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)


    df.columns = [c.strip().replace("  ", " ") for c in df.columns]

    if "Patient-ID" in df.columns:
        df = df.drop(columns=["Patient-ID"])


    if "Label" not in df.columns:
        raise ValueError("Expected a 'Label' column in the CSV.")

    df["Label"] = df["Label"].astype(str).str.strip()
    label_map = {"Non-Metastasis": 0, "Metastasis": 1}
    if not set(df["Label"].unique()).issubset(label_map.keys()):
        raise ValueError(f"Unexpected label values: {df['Label'].unique()}")

    y = df["Label"].map(label_map).values
    X = df.drop(columns=["Label"]).astype(float).values
    return X, y


def train_svm(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }

    return clf, scaler, metrics


def main():
    parser = argparse.ArgumentParser(description="Train non-linear SVM on random_uni.csv")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("/root/data/random_uni.csv"),
        help="Path to the CSV file.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("random_uni_svm.joblib"),
        help="Where to save the trained model and scaler.",
    )
    args = parser.parse_args()

    X, y = load_data(args.csv_path)
    clf, scaler, metrics = train_svm(X, y)

    print("Accuracy:", metrics["accuracy"])
    print("ROC AUC:", metrics["roc_auc"])
    print("Confusion matrix:\n", metrics["confusion_matrix"])
    print("Classification report:\n", metrics["classification_report"])

    args.model_out = args.model_out.resolve()
    joblib.dump({"model": clf, "scaler": scaler}, args.model_out)
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()

