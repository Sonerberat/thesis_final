import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def compute_stats(
    data_dir: str,
    resize: tuple[int, int] | None,
    batch_size: int,
    num_workers: int,
    max_images: int | None,
):
    tfs = []
    if resize is not None:
        tfs.append(transforms.Resize(resize))
    tfs.append(transforms.ToTensor())
    tfm = transforms.Compose(tfs)

    ds = datasets.ImageFolder(data_dir, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    sum_channels = torch.zeros(3, dtype=torch.double)
    sum_sq_channels = torch.zeros(3, dtype=torch.double)
    pixel_count = 0
    images_seen = 0
    h = w = None

    for x, _ in loader:
        if max_images is not None and images_seen >= max_images:
            break

        b, c, h, w = x.shape
        take = b
        if max_images is not None and images_seen + b > max_images:
            take = max_images - images_seen
            x = x[:take]
            b = take

        x = x.to(dtype=torch.double)
        pixels = b * h * w
        sum_channels += x.sum(dim=(0, 2, 3))
        sum_sq_channels += (x * x).sum(dim=(0, 2, 3))
        pixel_count += pixels
        images_seen += b

    if pixel_count == 0:
        raise RuntimeError("No images were processed; check the dataset path and contents.")

    mean = sum_channels / pixel_count
    std = torch.sqrt(sum_sq_channels / pixel_count - mean * mean)
    return {
        "images": images_seen,
        "size": (h, w),
        "mean": mean,
        "std": std,
    }


def main():
    p = argparse.ArgumentParser(description="Compute per-channel mean/std for UniKlinikum (or any ImageFolder) dataset.")
    p.add_argument(
        "--data-dir",
        type=str,
        default="/home/erensr/data/random_uniklinikum/train",
        help="Path to the ImageFolder split to analyze (e.g., train).",
    )
    p.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Resize to H W before stats (default: none, uses native size).",
    )
    p.add_argument("--batch-size", type=int, default=512, help="Batch size for loading images.")
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images for a quick estimate (default: use all).",
    )
    args = p.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    resize = tuple(args.resize) if args.resize is not None else None
    stats = compute_stats(
        data_dir=args.data_dir,
        resize=resize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images=args.max_images,
    )

    mean = stats["mean"].tolist()
    std = stats["std"].tolist()
    print(f"Images processed: {stats['images']}")
    print(f"Spatial size (after resize): {stats['size'][0]}x{stats['size'][1]}")
    print(f"Mean (full precision): {mean}")
    print(f"Std  (full precision): {std}")
    print("Copy-paste friendly:")
    print(f"unik_mean = ({mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f})")
    print(f"unik_std  = ({std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f})")


if __name__ == "__main__":
    main()
