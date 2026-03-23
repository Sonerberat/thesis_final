from __future__ import annotations

import argparse
import csv
from pathlib import Path


def find_tiff_files(dataset_dir: Path):
    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )


def save_tiff_names(dataset_dir: Path, output_csv: Path):
    tiff_files = find_tiff_files(dataset_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "relative_path", "split", "label"])

        for file_path in tiff_files:
            relative_path = file_path.relative_to(dataset_dir)
            parts = relative_path.parts
            split = parts[0] if len(parts) > 0 else ""
            label = parts[1] if len(parts) > 1 else ""
            writer.writerow([file_path.name, str(relative_path), split, label])

    return len(tiff_files)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save TIFF file names from a dataset into a CSV file."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/home/erensr/data/random_uniklinikum"),
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/home/erensr/ERTECnet/random_uniklinikum_tiff_names.csv"),
        help="Path to the CSV file that will be created.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    count = save_tiff_names(dataset_dir, output_csv)
    print(f"Saved {count} TIFF file names to {output_csv}")


if __name__ == "__main__":
    main()
