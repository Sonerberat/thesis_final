import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple


def load_svm_results(svm_results_path: Path, patient_id: str) -> Dict[str, float]:
    "Return SVM class probabilities for the requested patient."
    with svm_results_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("Patient-ID", "").strip() == patient_id:
                try:
                    prob_non_met = float(row["Prob_Non-Metastasis"])
                    prob_met = float(row["Prob_Metastasis"])
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"Invalid SVM results row for {patient_id!r}") from exc
                return {"prob_non_met": prob_non_met, "prob_stage4": prob_met}
    raise ValueError(f"Patient {patient_id!r} not found in {svm_results_path}")


def load_ertecnet_results(ertecnet_results_path: Path) -> List[Dict[str, Any]]:
    "Load ERTECNet rows and parse prob_non_met/prob_stage4 as float."
    rows: List[Dict[str, Any]] = []
    with ertecnet_results_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if "prob_non_met" not in row or "prob_stage4" not in row:
                raise ValueError(
                    "Columns 'prob_non_met' and 'prob_stage4' missing in "
                    f"{ertecnet_results_path}"
                )
            try:
                prob_non_met = float(row["prob_non_met"])
                prob_stage4 = float(row["prob_stage4"])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid prob_non_met/prob_stage4 entry in {ertecnet_results_path}"
                ) from exc
            new_row = dict(row)
            new_row["prob_non_met"] = prob_non_met
            new_row["prob_stage4"] = prob_stage4
            rows.append(new_row)
    if not rows:
        raise ValueError(f"No ERTECNet rows found in {ertecnet_results_path}")
    return rows


def average_ertecnet_probs(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    "Return mean probabilities across all ERTECNet rows."
    total_non_met = sum(row["prob_non_met"] for row in rows)
    total_stage4 = sum(row["prob_stage4"] for row in rows)
    count = len(rows)
    return {
        "prob_non_met": total_non_met / count,
        "prob_stage4": total_stage4 / count,
    }


def normalize_weights(
    weight_svm_results: float, weight_ertecnet_results: float
) -> Tuple[float, float]:
    total = weight_svm_results + weight_ertecnet_results
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return weight_svm_results / total, weight_ertecnet_results / total


def soft_vote_results(
    svm_results: Dict[str, float],
    ertecnet_results: Dict[str, float],
    weight_svm_results: float,
    weight_ertecnet_results: float,
) -> Dict[str, float]:
    return {
        "prob_non_met": svm_results["prob_non_met"] * weight_svm_results
        + ertecnet_results["prob_non_met"] * weight_ertecnet_results,
        "prob_stage4": svm_results["prob_stage4"] * weight_svm_results
        + ertecnet_results["prob_stage4"] * weight_ertecnet_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Soft vote between SVM results and averaged ERTECNet probabilities."
        )
    )
    parser.add_argument("patient_id", help="Patient-ID string, e.g. 'MM 00006'")
    parser.add_argument(
        "--svm-results-file",
        type=Path,
        default=Path("/home/erensr/ERTECnet/9.Soft_voting/output_svm.csv"),
        help="CSV with Patient-ID, Prob_Non-Metastasis, Prob_Metastasis columns",
    )
    parser.add_argument(
        "--ertecnet-results-file",
        type=Path,
        default=Path("/home/erensr/ERTECnet/9.Soft_voting/tile_new/21034834.csv"),
        help="CSV with prob_non_met and prob_stage4 columns for ERTECNet patches",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default="/home/erensr/ERTECnet/9.Soft_voting/final_soft_voting.csv",
        help="Optional path to save soft-vote results as CSV",
    )
    parser.add_argument(
        "--weights",
        nargs=2,
        type=float,
        default=(0.4205, 0.5795),
        metavar=("SVM_RESULTS", "ERTECNET_RESULTS"),
        help="Weights for SVM results and ERTECNet averages (will be normalized).",
    )

    args = parser.parse_args()

    svm_results = load_svm_results(args.svm_results_file, args.patient_id)
    ertecnet_rows = load_ertecnet_results(args.ertecnet_results_file)
    avg_ertecnet_results = average_ertecnet_probs(ertecnet_rows)
    weight_svm_results, weight_ertecnet_results = normalize_weights(
        args.weights[0], args.weights[1]
    )
    soft_probs = soft_vote_results(
        svm_results, avg_ertecnet_results, weight_svm_results, weight_ertecnet_results
    )
    predicted_label = (
        "Non-Metastasis"
        if soft_probs["prob_non_met"] >= soft_probs["prob_stage4"]
        else "Metastasis"
    )

    print(f"Patient ID: {args.patient_id}")
    print(f"ERTECNet patch count: {len(ertecnet_rows)}")
    print(
        "SVM results: "
        f"non_met={svm_results['prob_non_met']:.6f}, "
        f"stage4={svm_results['prob_stage4']:.6f}"
    )
    print(
        "ERTECNet avg: "
        f"non_met={avg_ertecnet_results['prob_non_met']:.6f}, "
        f"stage4={avg_ertecnet_results['prob_stage4']:.6f}"
    )
    print(
        "Normalized weights: "
        f"svm_results={weight_svm_results:.6f}, "
        f"ertecnet_results={weight_ertecnet_results:.6f}"
    )
    print(
        "Soft-voted probs: "
        f"non_met={soft_probs['prob_non_met']:.6f}, "
        f"stage4={soft_probs['prob_stage4']:.6f}"
    )
    print(f"Predicted label: {predicted_label}")

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        file_exists = args.output_file.exists()
        mode = "a" if file_exists else "w"
        with args.output_file.open(mode, newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "patient_id",
                    "svm_results_prob_non_met",
                    "svm_results_prob_stage4",
                    "ertecnet_results_avg_prob_non_met",
                    "ertecnet_results_avg_prob_stage4",
                    "weight_svm_results",
                    "weight_ertecnet_results",
                    "soft_prob_non_met",
                    "soft_prob_stage4",
                    "predicted_label",
                    "ertecnet_patch_count",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "patient_id": args.patient_id,
                    "svm_results_prob_non_met": svm_results["prob_non_met"],
                    "svm_results_prob_stage4": svm_results["prob_stage4"],
                    "ertecnet_results_avg_prob_non_met": avg_ertecnet_results["prob_non_met"],
                    "ertecnet_results_avg_prob_stage4": avg_ertecnet_results["prob_stage4"],
                    "weight_svm_results": weight_svm_results,
                    "weight_ertecnet_results": weight_ertecnet_results,
                    "soft_prob_non_met": soft_probs["prob_non_met"],
                    "soft_prob_stage4": soft_probs["prob_stage4"],
                    "predicted_label": predicted_label,
                    "ertecnet_patch_count": len(ertecnet_rows),
                }
            )
        print(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main()
