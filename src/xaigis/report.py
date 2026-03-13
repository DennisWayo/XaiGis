from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .utils import ensure_parent, load_json


def build_report(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg["paths"]
    report_path = ensure_parent(paths["report_md"])
    metrics_path = paths["metrics_json"]
    importance_path = paths["importance_csv"]

    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    importance_df = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()

    lines: list[str] = []
    lines.append("# XaiGis Run Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    lines.append("## Model Metrics")
    lines.append("")
    if metrics.get("models"):
        lines.append("| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for model, m in metrics["models"].items():
            lines.append(
                f"| {model} | {m.get('roc_auc', 0):.4f} | {m.get('pr_auc', 0):.4f} | "
                f"{m.get('precision', 0):.4f} | {m.get('recall', 0):.4f} | {m.get('f1', 0):.4f} |"
            )
    else:
        lines.append("No metrics found.")
    lines.append("")

    lines.append("## Top Features")
    lines.append("")
    if not importance_df.empty:
        for model in sorted(importance_df["model"].unique()):
            lines.append(f"### {model}")
            top = (
                importance_df[importance_df["model"] == model]
                .sort_values("importance", ascending=False)
                .head(10)
            )
            for _, row in top.iterrows():
                lines.append(f"- {row['feature']}: {row['importance']:.4f} ({row['method']})")
            lines.append("")
    else:
        lines.append("No feature importance file found.")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] saved: {report_path}")
    return {"report_md": str(report_path)}
