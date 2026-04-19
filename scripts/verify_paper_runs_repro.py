#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SHA256: dict[str, str] = {
    "paper_runs/configs/south_kazakhstan_region_paper_run.json": "d2349b6db19326a548552ecaed1d8469b40e54c0123f7fd35b16db9298512826",
    "paper_runs/configs/south_kaz_perf_mt24_pw6_qml.json": "06b5051a85a4902320d8a2c6ee894213f57e5411cb8188cd81781e1fb28cb671",
    "paper_runs/configs/south_kaz_qml_smoke_mt24_pw6.json": "a5b3f27d724e6ba8cb5a433d9d399ac71393b9a1c2a195a7dededd5384555e6d",
    "paper_runs/regions/south_kazakhstan_region.geojson": "9aa844c4e3718b28967c37f32f4c3cd7f7e5221095cfa81a53234e2b3906da9f",
    "paper_runs/regions/south_kazakhstan_region_labels.geojson": "ed1341403bd489984b4fd55fff699cc0c5a871c85d38275987fe768360119819",
    "paper_runs/copernicus/south_kazakhstan_region/search_20260319T070227Z.json": "c633fded6671d48502f49543f8e926cb0a843b77dd734bda0fab8f039c508bc3",
    "paper_runs/copernicus/south_kazakhstan_region/search_20260319T070227Z.csv": "5ae0e11ac8f76199e3b41c6c56f9ed9c52af86d10093983778eb7c73b59e79a1",
    "paper_runs/copernicus/south_kazakhstan_region/last_query_params.json": "fc3f8d68b942c094dfb52d9a7e3b571df0bee0fe9e1ac31b3f5349b536ccf33b",
    "paper_runs/runs/south_kazakhstan_region/artifacts/metrics.json": "a21e36e253769803a0837bac6bce69faf8103b4c2ce8dce50fa8d33ad49fcc09",
    "paper_runs/runs/south_kazakhstan_region/artifacts/paper_eval_metrics.json": "36430c56b5b6ab358a257caff871bf3c9cb86329930b1845c692d2f16847e777",
    "paper_runs/runs/south_kazakhstan_region/artifacts/paper_eval_folds.csv": "20b3855b377178615d2f77cbd0ff73d8167ac1463654d84e496db9725bb17828",
    "paper_runs/runs/south_kazakhstan_region/artifacts/leak_fit_summary.json": "20e11ae60bf36ba60c7c701d9c59bc048874f54006686df8f5be3d3d4e4aee84",
    "paper_runs/runs/south_kazakhstan_region/artifacts/leak_fit_folds.csv": "e1c02304a969718b42184f7b6758ad13aa794cd85404b8722980dfe24e2b0aab",
    "paper_runs/runs/south_kazakhstan_region/analysis/summary_additional_analysis.json": "7037c3ee5610f51f2d31bf6aeee31562b8e9f20acad15e822a28a530ed88c68b",
    "paper_runs/runs/south_kazakhstan_region/analysis/table_model_metrics_with_ci.csv": "7f013af5a146e0c541e53c42b8b6b0a6b7ee1256785c52ddb46cbda3cdf9e647",
    "paper_runs/runs/south_kazakhstan_region/analysis/table_operating_points.csv": "75e614de7974a0f8fbf3944c18a458c6e25a267d8e038c551f96191523a702c3",
    "paper_runs/runs/south_kazakhstan_region/analysis/table_pairwise_deltas_vs_rf.csv": "b486b9574aaa74d0b591f16a96e950d5d84f8d5edb661ffd17b849a91e4da620",
    "paper_runs/runs/south_kazakhstan_region/analysis/table_proxy_value_scene_threshold.csv": "ea3ff2178ad915dc763f1858bcf4c1d3b359a4e7845788e6c50358bdea820db6",
    "paper_runs/runs/south_kazakhstan_region/analysis/table_runtime_pareto.csv": "eb6d4eda784de3c036d1232cd08f075d58430889dc8191986a164bb8b8c528ea",
    "paper_runs/runs/south_kazakhstan_region/analysis_spatial/spatial_metrics.json": "44d5cf1552e070f342704d04d0f44f4e69c8e0d0583783fa105443af72ef2718",
    "paper_runs/runs/south_kazakhstan_region/analysis_spatial/table_spatial_metrics.csv": "5d7522b4e0ec2dc8e960bc4ee3b2c97454d5cd9361b22e39b4ed2e53a4db98af",
    "paper_runs/runs/south_kazakhstan_region/scenes/selected_scene_map_metrics.csv": "43f4b9f077ba6de03c056a52db597372fbdd3dcfd629122935b2000fc11de421",
    "paper_runs/runs/south_kazakhstan_region/scenes/selected_scene_map_metrics.json": "a026b234ec551bca6d4d78bb8207e692c0560d127640d69dd6d8ee72e1d78ef6",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes() -> int:
    failures = 0
    for rel_path, expected in EXPECTED_SHA256.items():
        p = PROJECT_ROOT / rel_path
        if not p.exists():
            print(f"[FAIL] missing: {rel_path}")
            failures += 1
            continue
        actual = sha256_file(p)
        if actual != expected:
            print(f"[FAIL] hash mismatch: {rel_path}")
            print(f"       expected={expected}")
            print(f"       actual  ={actual}")
            failures += 1
        else:
            print(f"[OK]   {rel_path}")
    return failures


def verify_config(strict_inputs: bool) -> int:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from xaigis.config import load_config

    failures = 0
    cfg_path = PROJECT_ROOT / "paper_runs/configs/south_kazakhstan_region_paper_run.json"
    cfg = load_config(cfg_path)
    paths = cfg.get("paths", {})

    expected_types = {
        "safe_zip": Path,
        "safe_dir": Path,
        "safe_zips": list,
        "safe_dirs": list,
        "geology_geojson": Path,
        "structures_geojson": type(None),
        "dem_tif": type(None),
    }

    for key, expected_type in expected_types.items():
        value = paths.get(key)
        if not isinstance(value, expected_type):
            print(
                f"[FAIL] config type: paths.{key} expected {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
            failures += 1
        else:
            print(f"[OK]   config type: paths.{key} -> {type(value).__name__}")

    required_runtime_inputs = ("safe_zip", "safe_dir", "geology_geojson")
    for key in required_runtime_inputs:
        value = paths.get(key)
        if isinstance(value, Path) and not value.exists():
            msg = f"paths.{key} missing on disk: {value}"
            if strict_inputs:
                print(f"[FAIL] {msg}")
                failures += 1
            else:
                print(f"[WARN] {msg}")

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify exact reproducibility artifacts under paper_runs/"
    )
    parser.add_argument(
        "--strict-inputs",
        action="store_true",
        help="Fail when runtime inputs (safe_zip/safe_dir/geology_geojson) are absent.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hash_failures = verify_hashes()
    config_failures = verify_config(strict_inputs=args.strict_inputs)
    total_failures = hash_failures + config_failures
    if total_failures:
        print(f"[repro] FAILED with {total_failures} issue(s)")
        return 1
    print("[repro] PASS: paper_runs snapshot is exact for the checked artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
