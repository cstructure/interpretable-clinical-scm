import pandas as pd
import numpy as np
import json
import ast
from datetime import datetime

def generate_json_report(input_csv, output_json):
    """
    Generates a JSON report from the causal inference results CSV.

    Args:
        input_csv (str): Path to the input CSV file.
        output_json (str): Path for the output JSON file.
    """
    df = pd.read_csv(input_csv)

    # Expected values from the challenge description
    expected_values = {
        1: {
            'expected_lower': 0.92,
            'expected_upper': 1.55,
            'expected_point_estimate': 1.19
        },
        2: {
            'expected_lower': 0.72,
            'expected_upper': 0.94,
            'expected_point_estimate': 0.82
        },
        3: {
            'expected_lower': 0.51,
            'expected_upper': 0.81,
            'expected_point_estimate': 0.64
        }
    }

    output_data = {
        "team_name": "Developer",
        "project_name": "claude",
        "model_name_version": "claude_po",
        "treatment": "STEROIDS",
        "timestamp": datetime.now().isoformat(),
        "outcome_formula": "DEATH ~ STEROIDS",
        "n_bootstrap": 500, # As per the example JSON
        "is_plr": False,
        "doubly_robust": False,
        "overall": {},
    }

    total_outlier_boots = 0
    weighted_rr_sum = 0
    total_n = 0

    for index, row in df.iterrows():
        severity = int(row['severity'])
        severity_key = f"severity_{severity}"
        
        # Parse rr_estimates
        try:
            rr_estimates = ast.literal_eval(row['rr_estimates'])
        except (ValueError, SyntaxError):
            rr_estimates = []

        point_estimates = np.array(rr_estimates)
        
        # Calculate outlier_boots
        expected = expected_values.get(severity, {})
        lower_bound = expected.get('expected_lower', 0)
        upper_bound = expected.get('expected_upper', 0)
        outlier_boots = np.sum((point_estimates < lower_bound) | (point_estimates > upper_bound))
        total_outlier_boots += outlier_boots

        # Format pt_estimates as a string
        pt_estimates_str = ",".join(map(str, point_estimates))

        output_data[severity_key] = {
            "risk_ratio": {
                "point_estimate": row['ate_rr'],
                "expected_point_estimate": expected.get('expected_point_estimate'),
                "confidence_interval": {
                    "lower": row['ci_rr_lower'],
                    "upper": row['ci_rr_upper'],
                    "expected_lower": lower_bound,
                    "expected_upper": upper_bound
                },
                "outlier_boots": int(outlier_boots),
                "pt_estimates": pt_estimates_str
            }
        }

        # For overall calculation
        weighted_rr_sum += row['ate_rr'] * row['n_total']
        total_n += row['n_total']

    # Calculate overall risk ratio
    overall_rr = weighted_rr_sum / total_n if total_n > 0 else 0
    output_data['overall'] = {
        "risk_ratio": {
            "point_estimate": overall_rr,
            "outlier_boots": int(total_outlier_boots)
        }
    }

    # Write to JSON file
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Successfully generated JSON report at {output_json}")

if __name__ == "__main__":
    generate_json_report('causal_inference_results.csv', 'claude_po_sdy1662_res.json')
