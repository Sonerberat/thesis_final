import argparse
from pathlib import Path

import joblib
import pandas as pd


LABEL_MAP_INV = {0: "Non-Metastasis", 1: "Metastasis"}


def load_features(csv_path: Path):
    df = pd.read_csv(csv_path)
    # Clean column names to match training-time preprocessing
    df.columns = [c.strip().replace("  ", " ") for c in df.columns]

    # Preserve an identifier if present
    patient_id = df["Patient-ID"] if "Patient-ID" in df.columns else None

    # Remove identifier column, leave only numeric features
    if "Patient-ID" in df.columns:
        df = df.drop(columns=["Patient-ID"])

    X = df.astype(float).values
    return patient_id, X


def predict(csv_path: Path, model_path: Path):
    payload = joblib.load(model_path)
    model = payload["model"]
    scaler = payload["scaler"]

    patient_id, X = load_features(csv_path)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)
    preds = model.predict(X_scaled)

    results = pd.DataFrame(
        {
            "Patient-ID": patient_id if patient_id is not None else range(len(preds)),
            "Predicted_Label": [LABEL_MAP_INV[p] for p in preds],
            "Prob_Non-Metastasis": proba[:, 0],
            "Prob_Metastasis": proba[:, 1],
        }
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict categories using trained random_uni SVM.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("data/new.csv"),
        help="Path to the CSV containing new samples (same schema as training, without Label).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/root/random_uni_svm.joblib"),
        help="Path to the trained model joblib.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional path to save predictions as CSV.",
    )
    args = parser.parse_args()

    results = predict(args.csv_path, args.model_path)
    print(results)

    if args.out_csv:
        out_path = args.out_csv.resolve()
        results.to_csv(out_path, index=False)
        print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()

