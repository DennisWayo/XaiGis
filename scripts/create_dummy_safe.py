#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.transform import from_origin


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    root = project_root / "data" / "S2_scene.SAFE"
    img_root = root / "GRANULE" / "L2A_T39TWN_A000000_20250101T000000" / "IMG_DATA"
    r10 = img_root / "R10m"
    r20 = img_root / "R20m"
    r60 = img_root / "R60m"
    r10.mkdir(parents=True, exist_ok=True)
    r20.mkdir(parents=True, exist_ok=True)
    r60.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    tile = "T39TWN_20250101T000000"

    specs = {
        "B01": ("60m", 20, 20),
        "B02": ("10m", 120, 120),
        "B03": ("10m", 120, 120),
        "B04": ("10m", 120, 120),
        "B05": ("20m", 60, 60),
        "B06": ("20m", 60, 60),
        "B07": ("20m", 60, 60),
        "B08": ("10m", 120, 120),
        "B8A": ("20m", 60, 60),
        "B09": ("60m", 20, 20),
        "B10": ("60m", 20, 20),
        "B11": ("20m", 60, 60),
        "B12": ("20m", 60, 60),
    }

    for band, (res, h, w) in specs.items():
        if band == "B08":
            base = np.linspace(0.1, 0.9, w, dtype=np.float32)[None, :].repeat(h, axis=0)
        elif band == "B04":
            base = np.linspace(0.9, 0.1, h, dtype=np.float32)[:, None].repeat(w, axis=1)
        else:
            base = rng.uniform(0.0, 1.0, size=(h, w)).astype(np.float32)
        arr = (base * 10000.0).astype(np.uint16)

        if res == "10m":
            out_dir = r10
            px = 10.0
        elif res == "20m":
            out_dir = r20
            px = 20.0
        else:
            out_dir = r60
            px = 60.0

        transform = from_origin(50.0, 47.5, px, px)
        profile = {
            "driver": "JP2OpenJPEG",
            "height": h,
            "width": w,
            "count": 1,
            "dtype": "uint16",
            "crs": "EPSG:4326",
            "transform": transform,
        }
        out_path = out_dir / f"{tile}_{band}_{res}.jp2"
        with rio.open(out_path, "w", **profile) as dst:
            dst.write(arr, 1)
        print(f"wrote {out_path}")

    print(f"\nDummy SAFE created at: {root}")


if __name__ == "__main__":
    main()
