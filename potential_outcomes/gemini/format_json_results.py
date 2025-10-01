import json
import re
import pandas as pd
from datetime import datetime
import ast

# --- Configuration ---
RESULTS_TEXT_FILE = 'gemini25_stratified_results.txt'
RESULTS_CSV_FILE = 'results/ate_results.csv'
OUTPUT_JSON_FILE = 'gemini25_stratified_results_sdy1662_boot500_res.json'

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

def parse_results_file(filepath):
    """Parses the text results file to extract key metrics."""
    with open(filepath, 'r') as f:
        content = f.read()

    results = {}
    severity_sections = content.split('--- Severity Level: ')

    for section in severity_sections[1:]:
        severity_level = int(section.split(' ')[0])
        
        # Using regex to find the floating point numbers and integers
        median_estimate = float(re.search(r'- Median Estimate: (\S+)', section).group(1))
        ci_search = re.search(r'- 95% CI: \[(\S+), (\S+)\]', section)
        lower_ci = float(ci_search.group(1))
        upper_ci = float(ci_search.group(2))
        outlier_boots = int(re.search(r'- Estimates outside benchmark CI: (\d+)', section).group(1))

        results[severity_level] = {
            'point_estimate': median_estimate,
            'confidence_interval': {
                'lower': lower_ci,
                'upper': upper_ci
            },
            'outlier_boots': outlier_boots
        }
    return results

def main():
    """Main function to generate the submission JSON."""
    # 1. Parse the text evaluation file
    try:
        parsed_data = parse_results_file(RESULTS_TEXT_FILE)
    except FileNotFoundError:
        print(f"Error: Input file '{RESULTS_TEXT_FILE}' not found.")
        return
    except Exception as e:
        print(f"Error parsing file: {e}")
        return

    # 2. Read the CSV for point estimates and overall risk ratio
    try:
        results_df = pd.read_csv(RESULTS_CSV_FILE)
    except FileNotFoundError:
        print(f"Error: CSV file '{RESULTS_CSV_FILE}' not found.")
        return

    # 3. Build the JSON structure
    submission = {
        "team_name": "Developer",
        "project_name": "gemini_po",
        "model_name_version": "gemini_po",
        "treatment": "STEROIDS",
        "timestamp": datetime.now().isoformat(),
        "outcome_formula": "DEATH ~ STEROIDS",
        "n_bootstrap": 500,
        "is_plr": False, # As per example
        "doubly_robust": False, # As per example
    }

    # Calculate overall risk ratio from the full dataset
    # This is a simplified approach; a true overall estimate would require a separate analysis
    overall_rr = results_df['ate_rr'].mean()
    submission['overall'] = {
        'risk_ratio': {
            'point_estimate': overall_rr,
            'outlier_boots': 0 # Placeholder, not calculated for overall
        }
    }

    # Add severity-specific data
    for level, data in parsed_data.items():
        severity_key = f"severity_{level}"
        expected_vals = EXPECTED_VALUES.get(level, {})
        
        # Get the bootstrap point estimates from the CSV
        pt_estimates_str = results_df[results_df['severity_level'] == level]['ate_rr_point_estimates'].iloc[0]
        # Convert string list to comma-separated string
        pt_estimates_list = ast.literal_eval(pt_estimates_str)
        pt_estimates_formatted = ','.join(map(str, pt_estimates_list))

        submission[severity_key] = {
            'risk_ratio': {
                'point_estimate': data['point_estimate'],
                'expected_point_estimate': expected_vals.get('point_estimate'),
                'confidence_interval': {
                    'lower': data['confidence_interval']['lower'],
                    'upper': data['confidence_interval']['upper'],
                    'expected_lower': expected_vals.get('lower_ci'),
                    'expected_upper': expected_vals.get('upper_ci'),
                },
                'outlier_boots': data['outlier_boots'],
                'pt_estimates': pt_estimates_formatted
            }
        }

    # 4. Write to JSON file
    with open(OUTPUT_JSON_FILE, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"Successfully created submission file: {OUTPUT_JSON_FILE}")

if __name__ == '__main__':
    main()
