from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import load_config, write_default_config

DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "safe_zip": "/path/to/S2_scene.SAFE.zip",
        "safe_dir": "/path/to/S2_scene.SAFE",
        "work_dir": "./outputs",
        "artifacts_dir": "./artifacts",
        "geology_geojson": "./geology/atyrau_targets.geojson",
        "feature_stack_tif": "./outputs/S2_feature_stack_10m.tif",
        "feature_names_json": "./outputs/feature_names.json",
        "label_tif": "./outputs/h2_label_poly_10m.tif",
        "dataset_npz": "./artifacts/Xy_dataset.npz",
        "dataset_csv": "./artifacts/Xy_dataset.csv",
        "models_dir": "./artifacts/models",
        "predictions_dir": "./artifacts/predictions",
        "metrics_json": "./artifacts/metrics.json",
        "importance_csv": "./artifacts/feature_importance.csv",
        "report_md": "./artifacts/report.md",
    },
    "features": {
        "band_order": [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B10",
            "B11",
            "B12",
        ],
        "indices": ["NDVI", "NDWI", "NDSI", "NBR", "BSI"],
        "texture_source": "NDVI",
        "texture_window": 7,
        "eps": 1e-6,
    },
    "dataset": {"tile_size": 256, "max_per_tile": 3000, "random_seed": 42},
    "training": {
        "test_size": 0.2,
        "random_seed": 42,
        "threshold": 0.8,
        "models": {"sgd": True, "rf": True, "xgb": True, "lgbm": True},
        "sgd": {"loss": "log_loss"},
        "rf": {"n_estimators": 200, "max_depth": None, "n_jobs": -1},
        "xgb": {"n_estimators": 100, "max_depth": 5},
        "lgbm": {"learning_rate": 0.1},
    },
    "prediction": {"tile_size": 512},
    "explain": {"sample_size": 3000, "use_shap_if_available": True},
}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "init-config":
        out = write_default_config(args.out, DEFAULT_CONFIG)
        print(f"[cli] default config written: {out}")
        return

    cfg = load_config(args.config)
    if args.command == "prepare":
        from .features import prepare_features

        prepare_features(cfg)
    elif args.command == "rasterize-labels":
        from .labels import rasterize_labels

        rasterize_labels(cfg)
    elif args.command == "sample-dataset":
        from .dataset import sample_dataset

        sample_dataset(cfg)
    elif args.command == "train":
        from .modeling import train_models

        train_models(cfg)
    elif args.command == "predict":
        from .modeling import predict_rasters

        predict_rasters(cfg)
    elif args.command == "explain":
        from .explain import explain_models

        explain_models(cfg)
    elif args.command == "report":
        from .report import build_report

        build_report(cfg)
    elif args.command == "run-all":
        from .dataset import sample_dataset
        from .explain import explain_models
        from .features import prepare_features
        from .labels import rasterize_labels
        from .modeling import predict_rasters, train_models
        from .report import build_report

        prepare_features(cfg)
        rasterize_labels(cfg)
        sample_dataset(cfg)
        train_models(cfg)
        predict_rasters(cfg)
        explain_models(cfg)
        build_report(cfg)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xaigis",
        description="XaiGis geospatial ML pipeline CLI",
    )
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init-config", help="Write default JSON config")
    p_init.add_argument("--out", default="configs/default.json", type=Path)

    for cmd in [
        "prepare",
        "rasterize-labels",
        "sample-dataset",
        "train",
        "predict",
        "explain",
        "report",
        "run-all",
    ]:
        p = sub.add_parser(cmd, help=f"Run {cmd} step")
        p.add_argument("--config", default="configs/default.json", type=Path)

    return parser


if __name__ == "__main__":
    main()
