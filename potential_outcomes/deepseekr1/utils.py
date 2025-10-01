import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def load_data(file_path):
    """Load and preprocess the person-time dataset"""
    df = pd.read_csv(file_path)
    print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
    
    # Detect columns with excessive missingness (>30%)
    missing_percent = df.isnull().mean() * 100
    high_missing_cols = missing_percent[missing_percent > 30].index.tolist()
    if high_missing_cols:
        print(f"Warning: Columns with >30% missing values: {high_missing_cols}")
        print("Consider excluding these columns from analysis")
    
    # Convert boolean columns
    bool_cols = ['STEROIDS', 'DEATH', 'DIABETES', 'CHF', 'COPD', 'CKD', 'ASTHMA']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Handle missing values
    print("\nMissing values before imputation:")
    print(df.isnull().sum())
    
    # Fill missing values by column type
    for col in df.columns:
        if df[col].isnull().any():
            # Handle numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                print(f"Filling {col} (numeric) NaN with median: {median_val:.2f}")
                df[col].fillna(median_val, inplace=True)
            # Handle non-numeric columns
            else:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else ''
                print(f"Filling {col} (non-numeric) NaN with mode: {mode_val}")
                df[col].fillna(mode_val, inplace=True)
    
    print("\nMissing values after imputation:")
    print(df.isnull().sum())
    
    if df.isnull().sum().sum() > 0:
        print("\nWarning: Some NaN values remain after imputation!")
        nan_cols = df.columns[df.isnull().any()].tolist()
        print(f"Columns with remaining NaN: {nan_cols}")
    else:
        print("\nNo missing values remain after imputation")
    
    return df


def detect_high_missingness(df, threshold=0.3):
    """Identify columns with high percentage of missing values"""
    missing_percent = df.isnull().mean()
    return missing_percent[missing_percent > threshold].index.tolist()


def prepare_covariates(df):
    """Prepare covariate matrix"""
    covariates = [
        'AGE', 'GENDER', 'BMI', 'DIABETES', 'CHF', 'COPD', 'CKD', 'ASTHMA',
        'SYSTOL_BP_MAX', 'DIASTOL_BP_MAX', 'O2SAT_MIN', 'RESP_RATE_MAX',
        'WBC_COUNT', 'LYMPHOCYTE_NO', 'CRP_LOG2', 'D_DIMER_LOG2', 'FERRITIN_LOG2'
    ]
    
    # Select only existing columns
    covariates = [c for c in covariates if c in df.columns]
    return df[covariates]


def calculate_ate(results):
    """Calculate risk difference and risk ratio from potential outcomes"""
    rd = results['E[Y₁]-E[Y₀]'][0]
    rr = results['E[Y₁]'][0] / results['E[Y₀]'][0]
    return rd, rr


def bootstrap_ci(df, func, n_bootstraps=500, alpha=0.05):
    """Calculate bootstrap confidence intervals"""
    estimates = []
    n = len(df)
    
    for _ in range(n_bootstraps):
        sample = df.sample(n, replace=True)
        rd, rr = func(sample)
        estimates.append((rd, rr))
    
    rd_estimates = [e[0] for e in estimates]
    rr_estimates = [e[1] for e in estimates]
    
    rd_ci = np.percentile(rd_estimates, [100*alpha/2, 100*(1-alpha/2)])
    rr_ci = np.percentile(rr_estimates, [100*alpha/2, 100*(1-alpha/2)])
    
    return rd_ci, rr_ci, [float(round(e, 2)) for e in rr_estimates]


def plot_diagnostics(treatment, propensity_scores, severity_level):
    """Generate diagnostic plots for overlap and balance"""
    plt.figure(figsize=(12, 5))
    
    # Propensity score distribution
    plt.subplot(121)
    sns.histplot(
        x=propensity_scores,
        hue=treatment,
        element="step",
        stat="density",
        common_norm=False
    )
    plt.title(f'Severity {severity_level}: Propensity Score Distribution')
    plt.xlabel('Propensity Score')
    
    # Covariate balance (standardized mean differences)
    plt.subplot(122)
    # Balance plot would be added after calculating SMD
    plt.title(f'Severity {severity_level}: Covariate Balance')
    
    plt.tight_layout()
    plt.savefig(f'diagnostics_severity_{severity_level}.png')
    plt.close()
