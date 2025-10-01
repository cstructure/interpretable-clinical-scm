#!/usr/bin/env python3
"""Generate JSON summary from stratified bootstrap results CSV.

Usage
-----
$ python create_json_summary.py o3_stratified_res_500boot.csv --out summary.json

Outputs the JSON to stdout and optionally writes to --out.
"""
from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np

# RCT benchmarks
RCT_CI: Dict[int, Tuple[float, float]] = {1: (0.92, 1.55), 2: (0.72, 0.94), 3: (0.51, 0.81)}
RCT_POINT: Dict[int, float] = {1: 1.19, 2: 0.82, 3: 0.64}

TEAM_NAME = "Developer"
PROJECT_NAME = "good2seed"
MODEL_VERSION = "good2seed Final2"
TREATMENT_VAR = "STEROIDS"
OUTCOME_FORMULA = "DEATH ~ STEROIDS"
N_BOOTSTRAP = 500


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create JSON summary from stratified results CSV.")
    p.add_argument("csv", type=Path, help="Input *_res_*.csv with boot_rr column.")
    p.add_argument("--out", type=Path, help="Optional path to write JSON artifact.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv)
    required = {"severity", "risk_ratio", "boot_rr"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}")

    # Accumulators
    overall_point_estimates: List[float] = []
    overall_outliers = 0

    summary: Dict[str, object] = {
        "team_name": TEAM_NAME,
        "project_name": PROJECT_NAME,
        "model_name_version": MODEL_VERSION,
        "treatment": TREATMENT_VAR,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "outcome_formula": OUTCOME_FORMULA,
        "n_bootstrap": N_BOOTSTRAP,
        "is_plr": False,
        "doubly_robust": False,
    }

    # Per-severity sections
    for _, row in df.iterrows():
        severity = int(row["severity"])
        rr_hat = float(row["risk_ratio"])
        boot_rr_vals = [float(x) for x in ast.literal_eval(row["boot_rr"])]

        # Bootstrap CI
        ci_lower, ci_upper = np.percentile(boot_rr_vals, [2.5, 97.5])

        # Outliers relative to RCT CI
        exp_low, exp_high = RCT_CI[severity]
        outliers = sum((x < exp_low) or (x > exp_high) for x in boot_rr_vals)

        # Correct direction metric not requested but could reuse

        severity_key = f"severity_{severity}"
        summary[severity_key] = {
            "risk_ratio": {
                "point_estimate": rr_hat,
                "expected_point_estimate": RCT_POINT[severity],
                "confidence_interval": {
                    "lower": ci_lower,
                    "upper": ci_upper,
                    "expected_lower": exp_low,
                    "expected_upper": exp_high,
                },
                "outlier_boots": outliers,
                "pt_estimates": ",".join(f"{x:.2f}" for x in boot_rr_vals),
            }
        }

        overall_point_estimates.append(rr_hat)
        overall_outliers += outliers

    # Overall stats (simple mean point estimate, sum of outliers)
    overall_point = float(np.mean(overall_point_estimates))
    summary["overall"] = {
        "risk_ratio": {"point_estimate": overall_point, "outlier_boots": overall_outliers}
    }

    json_str = json.dumps(summary, indent=2)
    print(json_str)

    if args.out:
        args.out.write_text(json_str)
        print(f"\nJSON written to {args.out}")


if __name__ == "__main__":
    main()
