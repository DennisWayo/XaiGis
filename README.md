[![CI](https://github.com/DennisWayo/XaiGis/actions/workflows/ci.yml/badge.svg)](https://github.com/DennisWayo/XaiGis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41598--026--41845--0-blue.svg)](https://doi.org/10.1038/s41598-026-41845-0)

# XaiGis Software

## Introduction

XaiGis is a config-driven geospatial machine learning software pipeline for natural hydrogen prospectivity mapping from Sentinel-2 imagery. It converts raw SAFE products into analysis-ready feature stacks, rasterized training labels, sampled pixel datasets, trained classification models, prediction GeoTIFFs, and explainability outputs.

The software is designed for reproducible, end-to-end execution from the command line, with each stage exposed as a dedicated CLI command (`prepare`, `train`, `predict`, `explain`, `report`) and orchestrated by JSON configuration. This makes it practical both for one-off research runs and repeatable operational workflows across new areas of interest.

Production-oriented implementation of the XaiGis workflow includes:
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

## Related Work Citation

If this codebase supports your work, please also cite the upstream scientific study:

```bibtex
@Article{Wayo2026,
  author  = {Wayo, Dennis Delali Kwesi and Goliatt, Leonardo and Hazlett, Randy and Fustic, Milovan and Leila, Mahmoud},
  title   = {Integrated pixel-wise remote sensing and explainable machine learning for natural hydrogen exploration in southeastern part of Pricaspian Basin, Western Kazakhstan},
  journal = {Scientific Reports},
  year    = {2026},
  month   = {Feb},
  day     = {26},
  issn    = {2045-2322},
  doi     = {10.1038/s41598-026-41845-0},
  url     = {https://doi.org/10.1038/s41598-026-41845-0}
}
```
