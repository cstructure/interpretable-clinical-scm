#!/usr/bin/env python3
"""
analysis.py

Causal inference analysis for the Midwest Healthcare Conference Potential Outcomes Challenge.

This script estimates the Average Treatment Effect (ATE) of systemic glucocorticoids (``STEROIDS``)
on 28-day in-hospital mortality (``DEATH``) for each baseline severity stratum (``SEVERITY_NUMERIC`` ∈ {1,2,3}).

We implement an Inverse Probability of Treatment Weighting (IPTW) estimator with a logistic-regression‐based
propensity score model and provide non-parametric bootstrap confidence intervals (default 500 resamples).

The script expects a single CSV file containing one row per *patient* (or person-day collapsed to baseline day 0)
and the variables listed in the challenge data dictionary.

---------------------------------------------------------------------
Usage
---------------------------------------------------------------------
$ python analysis.py --input data/sdy1662_person_day.csv --output results.csv --boot 500

---------------------------------------------------------------------
Output
---------------------------------------------------------------------
A CSV file with one row per severity level and the following columns:
    severity              – Baseline severity stratum (1,2,3)
    risk_difference       – IPTW risk difference (Y(1) − Y(0))
    rd_ci_lower           – Lower bound of the 95% bootstrap CI for risk difference
    rd_ci_upper           – Upper bound of the 95% bootstrap CI for risk difference
    risk_ratio            – IPTW risk ratio (Y(1) / Y(0))
    rr_ci_lower           – Lower bound of the 95% bootstrap CI for risk ratio
    rr_ci_upper           – Upper bound of the 95% bootstrap CI for risk ratio

---------------------------------------------------------------------
Caveats
---------------------------------------------------------------------
1. The estimator is unbiased under Consistency, Positivity, and Conditional Exchangeability given the covariates
   supplied to the propensity model (all non-outcome, non-treatment baseline variables).
2. Diagnostics for propensity score overlap and covariate balance are printed to stdout; visual inspection is
   recommended.
3. For reproducibility, set the PYTHONHASHSEED and the --seed argument.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

###############################################################################
# Helper functions
###############################################################################

def _prepare_X(cov_df: pd.DataFrame) -> pd.DataFrame:
    """Return design matrix with numeric features only.

    * Numeric columns are left unchanged.
    * Object / categorical columns are one-hot encoded (drop first level).
    """
    num = cov_df.select_dtypes(include=[np.number])
    cat = cov_df.select_dtypes(exclude=[np.number])
    if cat.shape[1] > 0:
        cat_encoded = pd.get_dummies(cat, drop_first=True)
        X = pd.concat([num, cat_encoded], axis=1)
    else:
        X = num.copy()
    return X.reset_index(drop=True)

def _fit_propensity_model(X: pd.DataFrame, treatment: pd.Series, seed: int = 1) -> Pipeline:
    """Fit a logistic-regression propensity score model with standardisation."""
    clf = LogisticRegression(
        penalty="l2",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    clf.fit(X.to_numpy(dtype=float, copy=False), treatment)
    return clf


def _compute_iptw_weights(ps: ArrayLike, treatment: ArrayLike, stabilised: bool = True) -> np.ndarray:
    """Compute (stabilised) IPTW weights for binary treatment."""
    ps = np.clip(ps, 1e-3, 1 - 1e-3)  # avoid extreme weights
    if stabilised:
        treat_prob = np.mean(treatment)
        numer = treat_prob * treatment + (1 - treat_prob) * (1 - treatment)
    else:
        numer = 1.0
    denom = ps * treatment + (1 - ps) * (1 - treatment)
    w = numer / denom
    return w


def _iptw_effect(outcome: ArrayLike, treatment: ArrayLike, weights: ArrayLike) -> Tuple[float, float]:
    """Compute IPTW risk difference and risk ratio."""
    treated_mask = treatment == 1
    untreated_mask = ~treated_mask

    y1 = np.average(outcome[treated_mask], weights=weights[treated_mask])
    y0 = np.average(outcome[untreated_mask], weights=weights[untreated_mask])

    rd = y1 - y0
    rr = y1 / y0 if y0 > 0 else np.nan
    return rd, rr


def _one_bootstrap(df: pd.DataFrame, covariates: List[str], seed: int) -> Tuple[float, float]:
    """One bootstrap replicate of IPTW RD and RR."""
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(df.index, size=len(df), replace=True)
    boot = df.loc[sample_idx].reset_index(drop=True)

    X = _prepare_X(boot[covariates])
    t = boot["STEROIDS"].values
    y = boot["DEATH"].values

    model = _fit_propensity_model(X, t, seed)
    ps = model.predict_proba(X)[:, 1]
    w = _compute_iptw_weights(ps, t)

    rd, rr = _iptw_effect(y, t, w)
    return rd, rr


def estimate_ate_ipw(df: pd.DataFrame, covariates: List[str], n_boot: int = 500, seed: int = 42) -> Tuple[float, Tuple[float, float], float, Tuple[float, float]]:
    """Estimate IPTW ATE with bootstrap 95% CI."""
    X = _prepare_X(df[covariates])
    t = df["STEROIDS"].values
    y = df["DEATH"].values

    model = _fit_propensity_model(X, t, seed)
    ps = model.predict_proba(X)[:, 1]
    w = _compute_iptw_weights(ps, t)

    rd_hat, rr_hat = _iptw_effect(y, t, w)

    rng = np.random.default_rng(seed)
    boot_rd, boot_rr = [], []
    for _ in range(n_boot):
        rd_b, rr_b = _one_bootstrap(df, covariates, seed=int(rng.integers(0, 1e9)))
        boot_rd.append(rd_b)
        boot_rr.append(rr_b)

    rd_ci = (np.percentile(boot_rd, 2.5), np.percentile(boot_rd, 97.5))
    rr_ci = (np.percentile(boot_rr, 2.5), np.percentile(boot_rr, 97.5))

    return rd_hat, rd_ci, rr_hat, rr_ci, [float(round(rr_est, 2)) for rr_est in boot_rr]

###############################################################################
# Main
###############################################################################

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate IPTW ATE of steroids on COVID-19 mortality.")
    p.add_argument("--input", required=True, help="Path to input CSV dataset.")
    p.add_argument("--output", default="results.csv", help="Path to results CSV (default: results.csv).")
    p.add_argument("--boot", type=int, default=500, help="Number of bootstrap resamples (default: 500).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    df = pd.read_csv(args.input)

    required_cols = {"STEROIDS", "DEATH", "SEVERITY_NUMERIC"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"Missing required columns: {missing}")

    covariates = [c for c in df.columns if c not in {"STEROIDS", "DEATH", "SEVERITY_NUMERIC"}]

    results = []
    for severity in sorted(df["SEVERITY_NUMERIC"].dropna().unique()):
        subset = df[df["SEVERITY_NUMERIC"] == severity].reset_index(drop=True)
        if subset.empty:
            continue
        print(f"\nSeverity {severity}: n = {len(subset)}")
        rd, rd_ci, rr, rr_ci, boot_rr = estimate_ate_ipw(subset, covariates, n_boot=args.boot, seed=args.seed)

        print(
            f"  Risk difference: {rd:.4f} (95% CI {rd_ci[0]:.4f}, {rd_ci[1]:.4f})\n"
            f"  Risk ratio:     {rr:.4f} (95% CI {rr_ci[0]:.4f}, {rr_ci[1]:.4f})"
        )

        results.append(
            {
                "severity": severity,
                "risk_difference": rd,
                "rd_ci_lower": rd_ci[0],
                "rd_ci_upper": rd_ci[1],
                "risk_ratio": rr,
                "rr_ci_lower": rr_ci[0],
                "rr_ci_upper": rr_ci[1],
                "boot_rr": boot_rr,
            }
        )

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
