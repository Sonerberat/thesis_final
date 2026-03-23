from pathlib import Path
from PIL import Image
import numpy as np
import shutil


tiles_dir = Path("/home/erensr/Pictures/roi/stage4_ROI/tiles_st4/tile_st4/190353630_HE")
# If you prefer to MOVE bad tiles instead of delete, set this:
move_bad_tiles = True
bad_tiles_dir = Path("/home/erensr/Pictures/roi/stage4_ROI/tiles_st4/mostly whites") / "190353630_HE mostly_white"

white_threshold = 220       # 0–255; lower if slide background is a bit darker
max_white_fraction = 0.60   # delete/move tiles with >60% white pixels


if move_bad_tiles:
    bad_tiles_dir.mkdir(exist_ok=True)

for tile_path in sorted(tiles_dir.glob("*.tif")):   # adjust extension if needed
    img = Image.open(tile_path).convert("RGB")
    arr = np.array(img)

    # define "white" = all channels above threshold
    white_mask = (
        (arr[..., 0] >= white_threshold) &
        (arr[..., 1] >= white_threshold) &
        (arr[..., 2] >= white_threshold)
    )
    frac_white = white_mask.mean()

    if frac_white > max_white_fraction:
        if move_bad_tiles:
            dest = bad_tiles_dir / tile_path.name
            shutil.move(str(tile_path), dest)
            print(f"MOVED:  {tile_path.name}  ({frac_white:.1%} white)")
        else:
            tile_path.unlink()
            print(f"DELETED: {tile_path.name}  ({frac_white:.1%} white)")
    else:
        print(f"KEPT:   {tile_path.name}  ({frac_white:.1%} white)")
