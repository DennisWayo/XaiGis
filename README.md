# XaiGis Software

Production-oriented implementation of the XaiGis workflow from the manuscript and notebook:
- Sentinel-2 SAFE ingestion and 10 m harmonization
- 22-feature stack generation (13 bands + 5 indices + 4 texture proxies)
- Polygon-to-raster label creation
- Pixel dataset extraction with tile-aware sampling
- Model training (SGD, RF, XGBoost, LightGBM)
- GeoTIFF probability and threshold masks
- Explainability outputs (model importances with SHAP fallback)
- Metrics and markdown reporting

## Quick Start

```bash
cd /path/to/XaiGis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
xaigis init-config --out configs/default.json
xaigis prepare --config configs/default.json
xaigis rasterize-labels --config configs/default.json
xaigis sample-dataset --config configs/default.json
xaigis train --config configs/default.json
xaigis predict --config configs/default.json
xaigis explain --config configs/default.json
xaigis report --config configs/default.json
```

## Notes

- Update `configs/default.json` paths as needed for your local inputs.
- The pipeline is designed for large rasters and uses tile/window processing where relevant.
- XGBoost/LightGBM are optional at runtime; if unavailable, those models are skipped with a warning.
