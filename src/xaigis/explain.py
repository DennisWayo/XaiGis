from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .utils import ensure_parent, load_json


def explain_models(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg["paths"]
    ecfg = cfg["explain"]
    dataset_npz = paths["dataset_npz"]
    models_dir = paths["models_dir"]
    out_csv = ensure_parent(paths["importance_csv"])

    if not dataset_npz.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_npz}")
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    data = np.load(dataset_npz)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.uint8)
    sample_size = min(int(ecfg.get("sample_size", 3000)), x.shape[0])
    idx = np.random.default_rng(42).choice(x.shape[0], size=sample_size, replace=False)
    x_sub = x[idx]
    y_sub = y[idx]

    feature_names = _load_feature_names(paths["feature_names_json"], x.shape[1])
    rows: list[dict[str, Any]] = []

    for model_path in sorted(models_dir.glob("*.joblib")):
        model_name = model_path.stem
        model = joblib.load(model_path)
        importance, method = _compute_importance(
            model=model,
            x_sub=x_sub,
            y_sub=y_sub,
            use_shap=bool(ecfg.get("use_shap_if_available", True)),
        )
        importance = np.maximum(np.asarray(importance).reshape(-1), 0.0)
        if importance.size != len(feature_names):
            raise ValueError(
                f"Importance size mismatch for {model_name}: "
                f"{importance.size} vs {len(feature_names)} feature names."
            )
        s = float(importance.sum())
        if s > 0:
            importance = importance / s

        for feat, imp in zip(feature_names, importance):
            rows.append(
                {
                    "model": model_name,
                    "feature": feat,
                    "importance": float(imp),
                    "method": method,
                }
            )
        print(f"[explain] {model_name}: method={method}")

    df = pd.DataFrame(rows).sort_values(["model", "importance"], ascending=[True, False])
    df.to_csv(out_csv, index=False)
    print(f"[explain] saved importance table: {out_csv}")
    return {"importance_csv": str(out_csv), "rows": int(df.shape[0])}


def _compute_importance(
    model: Any,
    x_sub: np.ndarray,
    y_sub: np.ndarray,
    use_shap: bool,
) -> tuple[np.ndarray, str]:
    if use_shap:
        shap_imp = _compute_shap_importance(model, x_sub)
        if shap_imp is not None:
            return shap_imp, "shap_mean_abs"

    estimator = _last_estimator(model)
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_), "native_feature_importances"
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        if coef.ndim == 2:
            coef = coef[0]
        return np.abs(coef), "abs_coefficients"

    result = permutation_importance(
        model,
        x_sub,
        y_sub,
        n_repeats=3,
        random_state=42,
        n_jobs=-1,
        scoring="average_precision",
    )
    return np.asarray(result.importances_mean), "permutation_importance"


def _compute_shap_importance(model: Any, x_sub: np.ndarray) -> np.ndarray | None:
    try:
        import shap
    except Exception:
        return None

    estimator = _last_estimator(model)
    model_for_shap = estimator
    x_input = x_sub

    if isinstance(model, Pipeline):
        transforms = model[:-1]
        x_input = transforms.transform(x_sub)

    if not hasattr(estimator, "predict_proba"):
        return None

    try:
        explainer = shap.TreeExplainer(model_for_shap)
        shap_values = explainer.shap_values(x_input)
    except Exception:
        return None

    values = None
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif hasattr(shap_values, "values"):
        values = shap_values.values
        if values.ndim == 3:
            values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]
    else:
        values = shap_values

    if values is None:
        return None
    values = np.asarray(values)
    if values.ndim != 2:
        return None
    return np.mean(np.abs(values), axis=0)


def _last_estimator(model: Any) -> Any:
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def _load_feature_names(path, n_features: int) -> list[str]:
    if path.exists():
        data = load_json(path)
        names = data.get("feature_names", [])
        if len(names) == n_features:
            return names
    return [f"f{i:02d}" for i in range(n_features)]
