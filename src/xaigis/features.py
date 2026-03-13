from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from scipy.ndimage import maximum_filter, minimum_filter, sobel, uniform_filter

from .utils import ensure_dir, ensure_parent, save_json

BAND_PATTERN = re.compile(r"_(B(?:0[1-9]|1[0-2]|8A))_(10m|20m|60m)\.jp2$", re.IGNORECASE)
RES_PRIORITY = {"10m": 0, "20m": 1, "60m": 2}
TEXTURE_NAMES = ["TEX_MEAN", "TEX_STD", "TEX_RANGE", "TEX_GRAD"]


def prepare_features(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg["paths"]
    fcfg = cfg["features"]
    band_order: list[str] = fcfg["band_order"]
    indices: list[str] = fcfg["indices"]
    eps = float(fcfg.get("eps", 1e-6))
    texture_window = int(fcfg.get("texture_window", 7))

    ensure_dir(paths["work_dir"])
    ensure_dir(paths["artifacts_dir"])
    safe_dir = _ensure_safe_dir(paths["safe_dir"], paths["safe_zip"], paths["work_dir"])
    band_paths = _discover_band_paths(safe_dir)

    missing = [b for b in band_order if b not in band_paths]
    if missing:
        print(f"[prepare] warning: missing bands in SAFE, zero-filling: {missing}")
    if "B08" not in band_paths:
        raise FileNotFoundError("Reference band B08 was not found.")

    with rio.open(band_paths["B08"]) as ref_src:
        height, width = ref_src.height, ref_src.width
        profile = ref_src.profile.copy()
        blockx, blocky = _valid_block_sizes(width, height)
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=len(band_order) + len(indices) + len(TEXTURE_NAMES),
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=blockx,
            blockysize=blocky,
            BIGTIFF="YES",
        )

    stack_tif = ensure_parent(paths["feature_stack_tif"])
    feature_names = list(band_order) + list(indices) + list(TEXTURE_NAMES)

    with rio.open(stack_tif, "w", **profile) as dst:
        print(f"[prepare] writing base bands to {stack_tif}")
        for i, band in enumerate(band_order, start=1):
            if band in band_paths:
                arr = _read_resampled(band_paths[band], height, width)
                arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
            else:
                arr = np.zeros((height, width), dtype=np.float32)
            dst.write(arr, i)
            dst.set_band_description(i, band)

    band_to_idx = {name: i + 1 for i, name in enumerate(feature_names)}
    with rio.open(stack_tif, "r+") as dst:
        print("[prepare] computing spectral indices")
        for idx_name in indices:
            out_idx = band_to_idx[idx_name]
            for _, window in dst.block_windows(1):
                out = _compute_index_window(dst, band_to_idx, idx_name, window, eps)
                dst.write(out.astype(np.float32), out_idx, window=window)
            dst.set_band_description(out_idx, idx_name)

        print("[prepare] computing texture proxies")
        ndvi = dst.read(band_to_idx["NDVI"]).astype(np.float32)
        texture_map = _compute_textures(ndvi, texture_window)
        for name, arr in texture_map.items():
            out_idx = band_to_idx[name]
            dst.write(np.nan_to_num(arr, nan=0.0).astype(np.float32), out_idx)
            dst.set_band_description(out_idx, name)

    save_json(paths["feature_names_json"], {"feature_names": feature_names})
    print(f"[prepare] feature names saved to {paths['feature_names_json']}")
    print(f"[prepare] completed stack with {len(feature_names)} features")

    return {
        "feature_stack_tif": str(stack_tif),
        "feature_count": len(feature_names),
        "feature_names_json": str(paths["feature_names_json"]),
    }


def _ensure_safe_dir(safe_dir: Path, safe_zip: Path, work_dir: Path) -> Path:
    if safe_dir.exists():
        return safe_dir
    if not safe_zip.exists():
        raise FileNotFoundError(f"SAFE dir not found and zip not found: {safe_dir} / {safe_zip}")
    extract_root = ensure_dir(work_dir / "SAFE")
    print(f"[prepare] extracting {safe_zip} -> {extract_root}")
    with zipfile.ZipFile(safe_zip, "r") as zf:
        zf.extractall(extract_root)
    candidates = sorted(extract_root.rglob("*.SAFE"))
    if not candidates:
        raise FileNotFoundError(f"No .SAFE directory found after extraction from {safe_zip}")
    return candidates[0]


def _discover_band_paths(safe_dir: Path) -> dict[str, Path]:
    selected: dict[str, tuple[str, Path]] = {}
    for jp2 in safe_dir.rglob("*.jp2"):
        m = BAND_PATTERN.search(jp2.name)
        if not m:
            continue
        band, res = m.group(1).upper(), m.group(2).lower()
        existing = selected.get(band)
        if existing is None or RES_PRIORITY[res] < RES_PRIORITY[existing[0]]:
            selected[band] = (res, jp2)
    out = {band: p for band, (_, p) in selected.items()}
    return out


def _read_resampled(path: Path, out_h: int, out_w: int) -> np.ndarray:
    with rio.open(path) as src:
        arr = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.bilinear,
        )
    return arr.astype(np.float32)


def _compute_index_window(
    dst: rio.io.DatasetWriter,
    band_to_idx: dict[str, int],
    idx_name: str,
    window: rio.windows.Window,
    eps: float,
) -> np.ndarray:
    if idx_name == "NDVI":
        b08 = dst.read(band_to_idx["B08"], window=window)
        b04 = dst.read(band_to_idx["B04"], window=window)
        return (b08 - b04) / (b08 + b04 + eps)
    if idx_name == "NDWI":
        b08 = dst.read(band_to_idx["B08"], window=window)
        b11 = dst.read(band_to_idx["B11"], window=window)
        return (b08 - b11) / (b08 + b11 + eps)
    if idx_name == "NDSI":
        b11 = dst.read(band_to_idx["B11"], window=window)
        b12 = dst.read(band_to_idx["B12"], window=window)
        return (b11 - b12) / (b11 + b12 + eps)
    if idx_name == "NBR":
        b08 = dst.read(band_to_idx["B08"], window=window)
        b12 = dst.read(band_to_idx["B12"], window=window)
        return (b08 - b12) / (b08 + b12 + eps)
    if idx_name == "BSI":
        b11 = dst.read(band_to_idx["B11"], window=window)
        b04 = dst.read(band_to_idx["B04"], window=window)
        b08 = dst.read(band_to_idx["B08"], window=window)
        b02 = dst.read(band_to_idx["B02"], window=window)
        num = (b11 + b04) - (b08 + b02)
        den = (b11 + b04) + (b08 + b02)
        return num / (den + eps)
    raise ValueError(f"Unsupported index requested in config: {idx_name}")


def _compute_textures(source: np.ndarray, window: int) -> dict[str, np.ndarray]:
    window = max(3, int(window))
    mean = uniform_filter(source, size=window, mode="nearest")
    sq_mean = uniform_filter(source * source, size=window, mode="nearest")
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
    mx = maximum_filter(source, size=window, mode="nearest")
    mn = minimum_filter(source, size=window, mode="nearest")
    tex_range = mx - mn
    gx = sobel(source, axis=0, mode="nearest")
    gy = sobel(source, axis=1, mode="nearest")
    grad = np.hypot(gx, gy)
    return {
        "TEX_MEAN": mean.astype(np.float32),
        "TEX_STD": std.astype(np.float32),
        "TEX_RANGE": tex_range.astype(np.float32),
        "TEX_GRAD": grad.astype(np.float32),
    }


def _valid_block_sizes(width: int, height: int) -> tuple[int, int]:
    def one(dim: int) -> int:
        candidate = min(256, dim)
        candidate = (candidate // 16) * 16
        if candidate < 16:
            return 16
        return candidate

    return one(width), one(height)
