import argparse
import random
import shutil
from pathlib import Path

def move_random_tifs(src: Path, dst: Path, count: int, seed: int | None):
    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    if seed is not None:
        random.seed(seed)

    # Collect available .tif files, excluding any that already exist in destination
    tif_files = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() == ".tif"]
    existing = {p.name for p in dst.iterdir() if p.is_file() and p.suffix.lower() == ".tif"}
    available = [p for p in tif_files if p.name not in existing]

    if len(available) < count:
        raise SystemExit(f"Requested {count} files but only {len(available)} available (not already in dest).")

    sample = random.sample(available, count)
    for i, src_file in enumerate(sample, 1):
        shutil.move(str(src_file), dst / src_file.name)  
        if i % 1000 == 0 or i == count:
            print(f"Moved {i}/{count}")

def main():
    parser = argparse.ArgumentParser(description="Move random .tif files.")
    parser.add_argument("--src", type=Path, default=Path("/home/erensr/data/random_uniklinikum/train/stage4"))
    parser.add_argument("--dst", type=Path, default=Path("/home/erensr/data/random_uniklinikum/test/stage4"))
    parser.add_argument("-n", "--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, help="Optional RNG seed for reproducibility")
    args = parser.parse_args()
    move_random_tifs(args.src.expanduser(), args.dst.expanduser(), args.count, args.seed)

if __name__ == "__main__":
    main()
