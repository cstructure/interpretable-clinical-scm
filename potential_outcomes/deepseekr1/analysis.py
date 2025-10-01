import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from econml.dml import DML
from econml.metalearners import TLearner
from utils import load_data, prepare_covariates, calculate_ate, bootstrap_ci, plot_diagnostics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Main analysis function
def analyze_severity(df, severity_level, output_file):
    """Run causal analysis for a specific severity level"""
    # Filter by severity level
    subset = df[df['SEVERITY_NUMERIC'] == severity_level].copy()
    
    # Prepare data
    y = subset['DEATH'].values  # Outcome: 28-day mortality
    T = subset['STEROIDS'].values  # Treatment: glucocorticoids
    X = prepare_covariates(subset)  # Covariates
    
    # Check for NaN values
    if np.isnan(X).any() or np.isnan(y).any() or np.isnan(T).any():
        print("Error: Input contains NaN values after preprocessing!")
        print("NaN in X:", np.isnan(X).any(axis=0))
        print("NaN in y:", np.isnan(y).any())
        print("NaN in T:", np.isnan(T).any())
        return
    
    # Split data for cross-fitting
    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
        X, T, y, test_size=0.2, random_state=42
    )
    
    # Initialize models (using Random Forests for demonstration)
    model_y = RandomForestRegressor(n_estimators=100, random_state=42)
    model_t = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Doubly Robust Learner (AIPW)
    estimator = DML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        cv=3,
        random_state=42,
        model_final=LinearRegression()
    )
    
    # Fit estimator
    estimator.fit(y_train, T_train, X=X_train)
    
    # Calculate treatment effects
    te = estimator.ate(X_test)
    ate = np.mean(te)
    
    # Get propensity scores for diagnostics
    propensity_scores = estimator.propensity_score(X_test)
    
    # Generate diagnostics
    plot_diagnostics(T_test, propensity_scores, severity_level)
    
    # Bootstrap confidence intervals
    def bootstrap_func(sample):
        # Simplified version for bootstrapping
        y_bs = sample['DEATH'].values
        T_bs = sample['STEROIDS'].values
        X_bs = prepare_covariates(sample)
        
        # Refit model on bootstrap sample
        estimator.fit(y_bs, T_bs, X=X_bs)
        te_bs = estimator.ate(X_bs)
        return np.mean(te_bs), 1 + np.mean(te_bs) / y_bs.mean()
    
    rd_ci, rr_ci, rr_estimates = bootstrap_ci(subset, bootstrap_func)
    
    # Save results
    results = pd.DataFrame({
        'severity_level': [severity_level],
        'risk_difference': [ate],
        'rd_lower_ci': [rd_ci[0]],
        'rd_upper_ci': [rd_ci[1]],
        'risk_ratio': [1 + ate / y.mean()],
        'rr_lower_ci': [rr_ci[0]],
        'rr_upper_ci': [rr_ci[1]],
        'rr_estimates': [rr_estimates]
    })
    
    results.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    df = load_data("sdy1662.csv")  # Actual data will be released on 2025-09-20
    
    # Create empty results file
    open('results.csv', 'w').close()
    
    # Analyze each severity level
    for severity in [1, 2, 3]:
        print(f"Analyzing severity level {severity}...")
        analyze_severity(df, severity, "results.csv")
    
    print("Analysis complete. Results saved to results.csv")
    print("Diagnostic plots saved as diagnostics_severity_*.png")
