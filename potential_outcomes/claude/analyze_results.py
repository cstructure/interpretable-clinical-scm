import pandas as pd
import numpy as np
import ast

def analyze_results(file_path):
    """
    Analyzes the causal inference results from the given CSV file.

    Args:
        file_path (str): The path to the CSV file.
    """
    df = pd.read_csv(file_path)

    severity_params = {
        1: {
            'lower_ci': 0.92,
            'upper_ci': 1.55,
            'point_estimate': 1.19
        },
        2: {
            'lower_ci': 0.72,
            'upper_ci': 0.94,
            'point_estimate': 0.82
        },
        3: {
            'lower_ci': 0.51,
            'upper_ci': 0.81,
            'point_estimate': 0.64
        }
    }

    for severity, params in severity_params.items():
        print(f"--- Severity Level {severity} ---")
        
        row = df[df['severity'] == severity].iloc[0]
        rr_estimates_str = row['rr_estimates']
        
        # Safely evaluate the string to a list
        try:
            rr_estimates = ast.literal_eval(rr_estimates_str)
        except (ValueError, SyntaxError):
            print(f"Could not parse rr_estimates for severity {severity}")
            continue

        point_estimates = np.array(rr_estimates)
        total_estimates = len(point_estimates)

        # 1) Calculate the number of point estimates that fall outside of the published confidence intervals
        outside_ci = np.sum((point_estimates < params['lower_ci']) | (point_estimates > params['upper_ci']))
        outside_ci_perc = (outside_ci / total_estimates) * 100 if total_estimates > 0 else 0
        print(f"Number of point estimates outside published CI: {outside_ci} ({outside_ci_perc:.2f}%)")

        # 2) Calculate the number of point estimates that point to the correct direction of effect
        if params['point_estimate'] > 1:
            correct_direction = np.sum(point_estimates > 1)
        else:
            correct_direction = np.sum(point_estimates < 1)
        correct_direction_perc = (correct_direction / total_estimates) * 100 if total_estimates > 0 else 0
        print(f"Number of point estimates in correct direction: {correct_direction} ({correct_direction_perc:.2f}%)")

        # 3) Calculate the median from the point estimate vector
        median_estimate = np.median(point_estimates)
        print(f"Median of point estimates: {median_estimate:.4f}")

        #4 Calculate the 95% CI
        ci = np.percentile(point_estimates, [2.5, 97.5])
        print(f"95% CI: {ci[0]:.4f} - {ci[1]:.4f}")
        print("\n")

if __name__ == "__main__":
    analyze_results('causal_inference_results.csv')
