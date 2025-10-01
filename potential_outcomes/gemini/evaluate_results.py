import pandas as pd
import numpy as np
import ast

# --- Configuration ---
RESULTS_FILE = 'results/ate_results.csv'

# Expected values from the challenge description
EXPECTED_VALUES = {
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

def evaluate_results(results_df):
    """Evaluates the simulation results against the expected benchmarks."""
    
    print("--- Evaluating Bootstrap Point Estimates vs. Published Benchmarks ---")

    for index, row in results_df.iterrows():
        severity_level = int(row['severity_level'])
        
        if severity_level not in EXPECTED_VALUES:
            print(f"\nSkipping severity level {severity_level} (no benchmark defined).")
            continue

        print(f"\n--- Severity Level: {severity_level} ---")

        # Safely parse the string representation of the list
        try:
            point_estimates = ast.literal_eval(row['ate_rr_point_estimates'])
        except (ValueError, SyntaxError) as e:
            print(f"  Error parsing point estimates: {e}")
            continue
        
        total_estimates = len(point_estimates)
        benchmarks = EXPECTED_VALUES[severity_level]
        
        # 1. Calculate the number of estimates outside the expected CI
        outside_ci_count = sum(1 for est in point_estimates if not (benchmarks['lower_ci'] <= est <= benchmarks['upper_ci']))
        
        # 2. Calculate the number of estimates pointing in the correct direction
        correct_direction_count = 0
        if benchmarks['point_estimate'] > 1:
            # Correct direction is > 1 (harm)
            correct_direction_count = sum(1 for est in point_estimates if est > 1)
            direction_label = "> 1 (harm)"
        else:
            # Correct direction is < 1 (benefit)
            correct_direction_count = sum(1 for est in point_estimates if est < 1)
            direction_label = "< 1 (benefit)"

        # 3. Calculate the median and 95% CI of the bootstrap estimates
        median_estimate = np.median(point_estimates)
        lower_ci = np.percentile(point_estimates, 2.5)
        upper_ci = np.percentile(point_estimates, 97.5)

        print(f"  Total bootstrap estimates: {total_estimates}")
        print(f"  Benchmark CI: [{benchmarks['lower_ci']}, {benchmarks['upper_ci']}]")
        print(f"  Benchmark Point Estimate: {benchmarks['point_estimate']}")
        print("\n  Evaluation Results:")
        print(f"  - Estimates outside benchmark CI: {outside_ci_count} / {total_estimates} ({outside_ci_count/total_estimates:.2%})")
        print(f"  - Estimates in correct direction ({direction_label}): {correct_direction_count} / {total_estimates} ({correct_direction_count/total_estimates:.2%})")
        print(f"  - Median Estimate: {median_estimate}")
        print(f"  - 95% CI: [{lower_ci}, {upper_ci}]")

def main():
    """Main function to load results and run evaluation."""
    try:
        results_df = pd.read_csv(RESULTS_FILE)
    except FileNotFoundError:
        print(f"Error: Results file not found at '{RESULTS_FILE}'.")
        print("Please ensure the analysis has been run and the file exists.")
        return
        
    evaluate_results(results_df)

if __name__ == '__main__':
    main()
