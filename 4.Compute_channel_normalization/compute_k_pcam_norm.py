import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


class KPCAMDataset:
    def __init__(self, data_dir: Path):
        labels_csv = data_dir / "train_labels.csv"
        train_dir = data_dir / "train"
        if not (labels_csv.is_file() and train_dir.is_dir()):
            raise FileNotFoundError(
                f"Expected train_labels.csv and train/ under {data_dir}."
            )

        samples = []
        with labels_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row.get("id")
                if not img_id:
                    continue
                path = train_dir / f"{img_id}.tif"
                if path.is_file():
                    samples.append(path)
        if not samples:
            raise RuntimeError("No training samples found after reading train_labels.csv.")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def iter_images(self, max_images: int | None, resize: tuple[int, int] | None):
        count = 0
        for path in self.samples:
            if max_images is not None and count >= max_images:
                break
            img = Image.open(path).convert("RGB")
            if resize is not None:
                img = img.resize((resize[1], resize[0]), Image.BILINEAR)
            yield img
            count += 1


def compute_stats(dataset, max_images: int | None, resize: tuple[int, int] | None):
    sum_channels = np.zeros(3, dtype=np.float64)
    sum_sq_channels = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    images_seen = 0
    h = w = None

    for img in dataset.iter_images(max_images=max_images, resize=resize):
        arr = np.asarray(img, dtype=np.float64) / 255.0
        h, w = arr.shape[:2]
        pixels = h * w
        sum_channels += arr.reshape(-1, 3).sum(axis=0)
        sum_sq_channels += (arr.reshape(-1, 3) ** 2).sum(axis=0)
        pixel_count += pixels
        images_seen += 1

    if pixel_count == 0:
        raise RuntimeError("No images were processed; check the dataset path and contents.")

    mean = sum_channels / pixel_count
    std = np.sqrt(sum_sq_channels / pixel_count - mean * mean)
    return {
        "images": images_seen,
        "size": (h, w),
        "mean": mean,
        "std": std,
    }


def main():
    p = argparse.ArgumentParser(description="Compute per-channel mean/std for k-pcam (train split).")
    p.add_argument(
        "--data-dir",
        type=str,
        default="/home/erensr/data/k-pcam",
        help="Path to the k-pcam dataset root (contains train/ and train_labels.csv).",
    )
    p.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Resize to H W before stats (default: none, uses native size).",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images for a quick estimate (default: use all).",
    )
    args = p.parse_args()

    resize = tuple(args.resize) if args.resize is not None else None
    ds = KPCAMDataset(Path(args.data_dir).expanduser())
    stats = compute_stats(
        dataset=ds,
        max_images=args.max_images,
        resize=resize,
    )

    mean = stats["mean"].tolist()
    std = stats["std"].tolist()
    print(f"Images processed: {stats['images']}")
    print(f"Spatial size (after resize): {stats['size'][0]}x{stats['size'][1]}")
    print(f"Mean (full precision): {mean}")
    print(f"Std  (full precision): {std}")
    print("Copy-paste friendly:")
    print(f"kpcam_mean = ({mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f})")
    print(f"kpcam_std  = ({std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f})")


if __name__ == "__main__":
    main()
