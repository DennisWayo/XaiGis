[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Build Status](https://img.shields.io/badge/ML-yes-green)
![Build Status](https://img.shields.io/badge/DL-yes-blue)
![Contributions](https://img.shields.io/badge/contributions-welcome-gold)
![GitHub issues](https://img.shields.io/github/issues/DennisWayo/hydrogen-mapping-atyrau)
![GitHub forks](https://img.shields.io/github/forks/DennisWayo/hydrogen-mapping-atyrau)
![GitHub stars](https://img.shields.io/github/stars/DennisWayo/hydrogen-mapping-atyrau)

# Hydrogen-Rich Reservoir Mapping using Machine Learning and Deep Learning

This repository contains the full implementation of a machine learning and deep learning framework for the detection and mapping of potential natural hydrogen reservoirs in Atyrau, Kazakhstan. We leverage Sentinel-2 satellite imagery, synthetic geochemical features, and explainable artificial intelligence (XAI) methods to generate high-resolution hydrogen prospectivity maps.

The project combines ensemble machine learning models and a customized U-Net convolutional neural network, validated through clustering and explainability methods such as Grad-CAM.

## Features

- **Remote Sensing Preprocessing:** NDVI, SWIR1 resampling, synthetic geochemical feature generation.

- **Machine Learning Pipeline:** Ensemble stacking using Random Forest, XGBoost, LightGBM, RidgeClassifier.

- **Deep Learning Segmentation:** Custom U-Net for pixel-wise hydrogen class prediction.

- **Explainability:** Grad-CAM interpretation of deep learning outputs.

- **Disagreement Analysis:** Raster difference maps and clustering (KMeans and DBSCAN) for anomaly detection.

- **Interactive Visualization:** Folium map overlays for results.

## Data

- Sentinel-2 Imagery: Copernicus Open Access Hub (2025 acquisition).

- Synthetic Geochemical Features: NDVI, SWIR1, Soil Temperature, Soil Moisture, Rock Type.

- Sample Data: Minimal dummy samples provided for reproducibility.

## 📦 Dataset Access

Due to file size limitations on GitHub, the complete geospatial training dataset for hydrogen-rich region prediction in Atyrau is hosted externally.

🔗 [Download Dataset from Google Drive](https://drive.google.com/file/d/1IbGIL9xZsFaANWRBj2-LCep6x_saU4Gn/view?usp=drive_link)

The dataset includes:
- Sentinel-2 raster bands (NDVI, SWIR1) preprocessed at 10m resolution
- Synthetic geochemical feature layers (soil moisture, temperature, depth)
- Machine learning ground truth labels (binary H₂ high/low classes)
- Full metadata and spatial reference files (CRS, GeoTIFF format)

Please cite this repository if you use the dataset for research purposes.

## Citation

If you use this dataset or the Python script, please cite:

```bash
@misc{wayo2025xaigis,
  author       = {Wayo, Dennis Delali Kwesi},
  title        = {XaiGis: Explainable Geospatial AI Pipeline for Natural Hydrogen Mapping in Atyrau},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15592017},
  url          = {https://doi.org/10.5281/zenodo.15592017}
}
```

## Acknowledgements

- Copernicus Sentinel-2 Data.

- Google Colab for computational resources.

- PyTorch, Scikit-learn, Rasterio libraries.
