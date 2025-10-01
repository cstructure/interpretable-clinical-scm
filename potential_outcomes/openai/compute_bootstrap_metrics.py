#!/usr/bin/env python3
"""Compute bootstrap diagnostics for risk-ratio estimates.

Given a CSV produced by analysis.py that contains a column `boot_rr` (stringified
list of bootstrap risk-ratio point estimates) for each baseline severity level,
this script calculates:

1. The number of bootstrap estimates that fall outside the *published* (RCT)
   95 % confidence interval, supplied in the script below.
2. The number of bootstrap estimates that indicate the *correct* direction of
   effect, i.e. greater than 1 if the expected point estimate > 1 (harmful) or
   less than 1 if the expected point estimate < 1 (beneficial).
3. The median bootstrap risk ratio.

Usage
-----
$ python compute_bootstrap_metrics.py o3_stratified_res_500boot.csv

Outputs a small summary table to stdout.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Published RCT benchmarks provided by the user
# -----------------------------------------------------------------------------
RCT_CI: Dict[int, Tuple[float, float]] = {
    1: (0.92, 1.55),
    2: (0.72, 0.94),
    3: (0.51, 0.81),
}
RCT_POINT: Dict[int, float] = {
    1: 1.19,
    2: 0.82,
    3: 0.64,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap diagnostics for RR estimates.")
    p.add_argument("csv", type=Path, help="Path to *_res_*.csv file containing boot_rr column.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv)
    required_cols = {"severity", "boot_rr"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns {required_cols}")

    rows = []
    for _, row in df.iterrows():
        severity = int(row["severity"])
        boot_rr_str: str = row["boot_rr"]
        try:
            rr_vals = [float(x) for x in ast.literal_eval(boot_rr_str)]
        except (SyntaxError, ValueError):
            raise ValueError(f"Could not parse boot_rr list for severity {severity}")

        lower_ci, upper_ci = RCT_CI[severity]
        expected_point = RCT_POINT[severity]

        # 1. Count outside CI
        outside_ci = sum((rr < lower_ci) or (rr > upper_ci) for rr in rr_vals)

        # 2. Correct direction
        if expected_point > 1:
            correct_dir = sum(rr > 1 for rr in rr_vals)
        else:
            correct_dir = sum(rr < 1 for rr in rr_vals)

        # 3. Median
        median_rr = float(pd.Series(rr_vals).median())

        ci_lower = float(np.percentile(rr_vals, 2.5))
        ci_upper = float(np.percentile(rr_vals, 97.5))

        rows.append(
            {
                "severity": severity,
                "n_boot": len(rr_vals),
                "outside_CI": outside_ci,
                "correct_direction": correct_dir,
                "median_rr": median_rr,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("severity")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
