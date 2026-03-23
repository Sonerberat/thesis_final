from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Sequence

import torch
from PIL import Image
from torchvision import transforms

from ERTECNet_final_edition import RTECNet


# Normalization used during training for random_uniklinikum
_UNIK_MEAN = (0.740058, 0.530798, 0.684085)
_UNIK_STD = (0.181236, 0.226015, 0.167639)


def _default_dataset_root() -> Path | None:
    candidate = Path("/home/erensr/data/random_uniklinikum")
    return candidate if candidate.exists() else None


def infer_class_names(dataset_root: Path | None, fallback_count: int):

    if dataset_root:
        for sub in ("train", "."): 
            root = dataset_root / sub
            if root.exists() and root.is_dir():
                names = sorted([p.name for p in root.iterdir() if p.is_dir()])
                if len(names) == fallback_count and names:
                    return names

    return [f"class_{i}" for i in range(fallback_count)]


def build_model(ckpt_path: Path, device: torch.device):
    """Rebuild the model using checkpoint metadata and load weights."""
    ckpt = torch.load(ckpt_path, map_location=device,weights_only=True)
    state = ckpt["state_dict"]


    num_classes = state["esn.W_out"].shape[1]
    first_conv = state.get("cnn.stem.0.weight")
    in_channels = first_conv.shape[1] if first_conv is not None else 3

    ckpt_args = ckpt.get("args", {})
    image_size = tuple(ckpt_args.get("image_size", (96, 96)))
    esn_cfg = ckpt.get("esn_cfg", {})

    overrides = {}
    for key in ("L", "S", "neurons_per_deep", "ridge_lambda"):
        if key in esn_cfg:
            overrides[key] = esn_cfg[key]

    model = RTECNet(
        in_channels=in_channels,
        num_classes=num_classes,
        image_hw=image_size,
        esn_cfg_overrides=overrides,
        device=device,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.esn.w_out_fitted = True  
    model.eval()
    model.image_size = image_size  

    # Derive class names
    env_root = Path(os.environ["RANDOM_UNIKLINIKUM_DATASET_DIR"]) if os.environ.get("RANDOM_UNIKLINIKUM_DATASET_DIR") else None
    dataset_root_arg = ckpt_args.get("dataset_root")
    dataset_root = (
        Path(dataset_root_arg).expanduser()
        if dataset_root_arg
        else env_root or _default_dataset_root()
    )
    if dataset_root and dataset_root == Path("./data"):
        default_unik = _default_dataset_root()
        if default_unik:
            dataset_root = default_unik

    class_names = infer_class_names(dataset_root, num_classes)
    return model, class_names


def make_transform(image_size: Sequence[int]):
    return transforms.Compose(
        [
            transforms.Resize(tuple(image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_UNIK_MEAN, std=_UNIK_STD),
        ]
    )


def load_image(path: Path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def predict(model: RTECNet, class_names: Sequence[str], images: List[Path], csv_path: Path | None = None):
    device = next(model.parameters()).device
    image_size = getattr(model, "image_size", None)
    if image_size is None:
        raise RuntimeError("Model is missing image_size information needed for resizing.")
    tfm = make_transform(image_size)

    csv_file = None
    writer = None
    if csv_path:
        csv_path = csv_path.expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")
        writer = csv.writer(csv_file)
        header = ["image_path", "pred_label"] + [f"prob_{cls}" for cls in class_names]
        writer.writerow(header)
        csv_file.flush()
        print(f"[csv] Writing probabilities to {csv_path} (streaming)")

    for img_path in images:
        if not img_path.exists():
            print(f"[warn] Missing file: {img_path}")
            continue

        img = load_image(img_path)
        x = tfm(img).unsqueeze(0)
        if device.type == "cuda":
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            x = x.to(device)

        with torch.inference_mode():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(probs.argmax())
            pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

        print(f"\n{img_path}:")
        for cls, p in zip(class_names, probs):
            print(f"  {cls:>15}: {p*100:6.2f}%")
        if writer:
            writer.writerow([img_path.as_posix(), pred_label] + probs.tolist())
            csv_file.flush()

    if csv_file:
        csv_file.close()
        print(f"\nSaved probabilities to {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict class probabilities with model_random_uni.pt.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/erensr/ERTECnet/unioutput/model_uni.pt"),
        help="Path to the trained checkpoint (.pt)",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to save probabilities as CSV (one row per image).",
    )
    parser.add_argument(
        "--image",
        dest="images",
        type=Path,
        nargs="+",
        required=True,
        help="Image file(s) or directories to classify. Directories will be scanned for common image extensions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda|cpu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, class_names = build_model(args.checkpoint, device)
    paths: List[Path] = []
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    for p in [p.expanduser().resolve() for p in args.images]:
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in exts:
                    paths.append(f)
        else:
            paths.append(p)

    if not paths:
        raise SystemExit("No image files found. Please provide image paths or directories containing images.")

    predict(
        model,
        class_names,
        paths,
        csv_path=args.csv_out,
    )


if __name__ == "__main__":
    main()
