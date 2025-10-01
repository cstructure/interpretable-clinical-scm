#!/usr/bin/env python3
"""
Causal Inference Analysis: Estimating the Effect of Steroids on COVID-19 In-Hospital Mortality
=============================================================================
This script implements potential outcomes framework to estimate the causal effect
of systemic glucocorticoids on 28-day in-hospital survival among COVID-19 patients,
stratified by baseline severity level.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class CausalInferenceAnalysis:
    """
    Main class for causal inference analysis using potential outcomes framework.
    
    Causal Assumptions:
    1. Consistency: Treatment received = potential treatment (SUTVA)
    2. Positivity: 0 < P(A=1|C) < 1 for all C
    3. Conditional Exchangeability: Y(a) ⊥ A | C
    """
    
    def __init__(self):
        # Define confounders based on clinical relevance and data availability
        self.confounders = [
            # Demographics
            'AGE', 'GENDER', 'BMI',
            # Comorbidities
            'DIABETES', 'CKD', 'CHF', 'COPD', 'ASTHMA', 'CANCER_FLAG',
            'CORONARY_ARTERY_DISEASE', 'ATRIAL_FIBRILLATION',
            'ALCOHOLIC_NONALCOHOLIC_LIVER_DISEASE', 'CHRONIC_VIRAL_HEPATITIS',
            # Lab values (baseline/admission)
            'ALBUMIN', 'CREATININE_LOG2', 'BUN', 'CRP_LOG2', 'D_DIMER_LOG2',
            'FERRITIN_LOG2', 'LDH_LOG2', 'LYMPHOCYTE_NO', 'NEUTROPHIL_NO',
            'PLATELET', 'WBC_COUNT', 'HEMOGLOBIN', 'O2SAT_MIN',
            # Clinical markers
            'TEMP_MAX', 'RESP_RATE_MAX', 'SYSTOL_BP_MAX', 'DIASTOL_BP_MAX'
        ]
        
        self.results = {}
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the person-day dataset."""
        print("Loading data...")
        df = pd.read_csv(filepath)
        
        # Filter for Study Day 0 (baseline) for treatment assignment
        # This ensures we capture treatment within 24h of admission
        df_baseline = df[df['STUDY_DAY'] == 0].copy()
        
        # Get outcome at Day 28 or last available day
        df_outcome = df.groupby('SUBJECT_ACCESSION').agg({
            'DEATH': 'max',  # Death at any point
            'STUDY_DAY': 'max'  # Last observed day
        }).reset_index()
        
        # Merge baseline and outcome data
        df_analysis = pd.merge(
            df_baseline,
            df_outcome[['SUBJECT_ACCESSION', 'DEATH']].rename(columns={'DEATH': 'DEATH_28D'}),
            on='SUBJECT_ACCESSION',
            how='left'
        )
        
        # Handle missing values
        for col in self.confounders:
            if col in df_analysis.columns:
                if df_analysis[col].dtype in ['float64', 'int64']:
                    df_analysis[col].fillna(df_analysis[col].median(), inplace=True)
                else:
                    df_analysis[col].fillna(df_analysis[col].mode()[0], inplace=True)
        
        # Create binary gender variable
        if 'GENDER' in df_analysis.columns:
            df_analysis['GENDER'] = (df_analysis['GENDER'] == 'Male').astype(int)
        
        print(f"Total patients: {len(df_analysis)}")
        print(f"Treated (steroids): {df_analysis['STEROIDS'].sum()}")
        print(f"Deaths (28-day): {df_analysis['DEATH_28D'].sum()}")
        
        return df_analysis
    
    def check_positivity(self, df, severity):
        """Check positivity assumption."""
        df_sev = df[df['SEVERITY_NUMERIC'] == severity]
        prop_score = df_sev.groupby('STEROIDS').size() / len(df_sev)
        
        print(f"\nSeverity {severity} - Treatment proportions:")
        print(f"  No steroids: {prop_score.get(0, 0):.3f}")
        print(f"  Steroids: {prop_score.get(1, 0):.3f}")
        
        if prop_score.get(0, 0) < 0.05 or prop_score.get(1, 0) < 0.05:
            print("  WARNING: Potential positivity violation!")
    
    def estimate_propensity_scores(self, df, severity):
        """Estimate propensity scores using logistic regression."""
        df_sev = df[df['SEVERITY_NUMERIC'] == severity].copy()
        
        # Prepare features
        available_confounders = [c for c in self.confounders if c in df_sev.columns]
        X = df_sev[available_confounders]
        A = df_sev['STEROIDS']
        
        # Fit propensity score model
        ps_model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
        ps_model.fit(X, A)
        
        # Get propensity scores
        ps = ps_model.predict_proba(X)[:, 1]
        df_sev['propensity_score'] = ps
        
        return df_sev, ps_model
    
    def create_weights(self, df_sev):
        """Create inverse probability of treatment weights (IPTW)."""
        ps = df_sev['propensity_score']
        A = df_sev['STEROIDS']
        
        # Stabilized weights
        p_treat = A.mean()
        weights = np.where(A == 1, p_treat / ps, (1 - p_treat) / (1 - ps))
        
        # Truncate extreme weights
        weights = np.clip(weights, np.percentile(weights, 1), np.percentile(weights, 99))
        
        return weights
    
    def iptw_estimation(self, df, severity):
        """Inverse Probability of Treatment Weighting estimation."""
        df_sev, ps_model = self.estimate_propensity_scores(df, severity)
        weights = self.create_weights(df_sev)
        
        # Weighted outcomes
        Y = df_sev['DEATH_28D']
        A = df_sev['STEROIDS']
        
        # Risk under treatment
        risk_1 = np.sum(weights * A * Y) / np.sum(weights * A)
        
        # Risk under control
        risk_0 = np.sum(weights * (1 - A) * Y) / np.sum(weights * (1 - A))
        
        # ATE
        ate_rd = risk_1 - risk_0  # Risk difference
        ate_rr = risk_1 / risk_0 if risk_0 > 0 else np.nan  # Risk ratio
        
        return {
            'risk_1': risk_1,
            'risk_0': risk_0,
            'ate_rd': ate_rd,
            'ate_rr': ate_rr,
            'ps_model': ps_model,
            'df_sev': df_sev,
            'weights': weights
        }
    
    def g_computation(self, df, severity):
        """G-computation (outcome regression) estimation."""
        df_sev = df[df['SEVERITY_NUMERIC'] == severity].copy()
        
        # Prepare features
        available_confounders = [c for c in self.confounders if c in df_sev.columns]
        X = df_sev[available_confounders].copy()
        A = df_sev['STEROIDS']
        Y = df_sev['DEATH_28D']
        
        # Add treatment to features
        X['STEROIDS'] = A
        
        # Fit outcome model
        outcome_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        outcome_model.fit(X, Y)
        
        # Predict outcomes under both treatments
        X1 = X.copy()
        X1['STEROIDS'] = 1
        Y1_pred = outcome_model.predict_proba(X1)[:, 1]
        
        X0 = X.copy()
        X0['STEROIDS'] = 0
        Y0_pred = outcome_model.predict_proba(X0)[:, 1]
        
        # Calculate risks
        risk_1 = Y1_pred.mean()
        risk_0 = Y0_pred.mean()
        
        # ATE
        ate_rd = risk_1 - risk_0
        ate_rr = risk_1 / risk_0 if risk_0 > 0 else np.nan
        
        return {
            'risk_1': risk_1,
            'risk_0': risk_0,
            'ate_rd': ate_rd,
            'ate_rr': ate_rr,
            'outcome_model': outcome_model,
            'Y1_pred': Y1_pred,
            'Y0_pred': Y0_pred
        }
    
    def aipw_estimation(self, df, severity):
        """Augmented Inverse Probability Weighting (doubly robust) estimation."""
        # Get IPTW components
        iptw_res = self.iptw_estimation(df, severity)
        df_sev = iptw_res['df_sev']
        ps = df_sev['propensity_score']
        
        # Get g-computation components
        g_comp_res = self.g_computation(df, severity)
        Y1_pred = g_comp_res['Y1_pred']
        Y0_pred = g_comp_res['Y0_pred']
        
        # Observed values
        A = df_sev['STEROIDS'].values
        Y = df_sev['DEATH_28D'].values
        
        # AIPW estimator
        psi_1 = Y1_pred + (A * (Y - Y1_pred)) / ps
        psi_0 = Y0_pred + ((1 - A) * (Y - Y0_pred)) / (1 - ps)
        
        risk_1 = psi_1.mean()
        risk_0 = psi_0.mean()
        
        ate_rd = risk_1 - risk_0
        ate_rr = risk_1 / risk_0 if risk_0 > 0 else np.nan
        
        return {
            'risk_1': risk_1,
            'risk_0': risk_0,
            'ate_rd': ate_rd,
            'ate_rr': ate_rr,
            'psi_1': psi_1,
            'psi_0': psi_0
        }
    
    def bootstrap_confidence_intervals(self, df, severity, method='aipw', n_bootstrap=500):
        """Calculate confidence intervals using bootstrap."""
        n = len(df[df['SEVERITY_NUMERIC'] == severity])
        results_boot = {'ate_rd': [], 'ate_rr': []}
        
        for i in range(n_bootstrap):
            # Resample
            idx = np.random.choice(n, n, replace=True)
            df_boot = df[df['SEVERITY_NUMERIC'] == severity].iloc[idx]
            
            try:
                if method == 'iptw':
                    res = self.iptw_estimation(df_boot, severity)
                elif method == 'g_comp':
                    res = self.g_computation(df_boot, severity)
                elif method == 'aipw':
                    res = self.aipw_estimation(df_boot, severity)
                
                results_boot['ate_rd'].append(res['ate_rd'])
                results_boot['ate_rr'].append(res['ate_rr'])
            except:
                continue
        
        # Calculate percentile CIs
        ci_rd = np.percentile(results_boot['ate_rd'], [2.5, 97.5])
        ci_rr = np.percentile(results_boot['ate_rr'], [2.5, 97.5])
        rr_estimates = [float(round(x, 2)) for x in results_boot['ate_rr']]
        
        return {'ci_rd': ci_rd, 'ci_rr': ci_rr, 'rr_estimates': rr_estimates}
    
    def check_balance(self, df_sev, weights=None):
        """Check covariate balance before and after weighting."""
        available_confounders = [c for c in self.confounders if c in df_sev.columns]
        
        balance_results = []
        
        for var in available_confounders:
            if df_sev[var].dtype in ['float64', 'int64']:
                # Unweighted
                mean_1 = df_sev[df_sev['STEROIDS'] == 1][var].mean()
                mean_0 = df_sev[df_sev['STEROIDS'] == 0][var].mean()
                std_pooled = np.sqrt((df_sev[df_sev['STEROIDS'] == 1][var].var() + 
                                     df_sev[df_sev['STEROIDS'] == 0][var].var()) / 2)
                smd_unweighted = (mean_1 - mean_0) / std_pooled if std_pooled > 0 else 0
                
                # Weighted
                if weights is not None:
                    mean_1_w = np.average(df_sev[df_sev['STEROIDS'] == 1][var], 
                                         weights=weights[df_sev['STEROIDS'] == 1])
                    mean_0_w = np.average(df_sev[df_sev['STEROIDS'] == 0][var], 
                                         weights=weights[df_sev['STEROIDS'] == 0])
                    smd_weighted = (mean_1_w - mean_0_w) / std_pooled if std_pooled > 0 else 0
                else:
                    smd_weighted = np.nan
                
                balance_results.append({
                    'variable': var,
                    'smd_unweighted': smd_unweighted,
                    'smd_weighted': smd_weighted
                })
        
        return pd.DataFrame(balance_results)
    
    def plot_diagnostics(self, df_sev, weights, severity, balance_df):
        """Create diagnostic plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Propensity score distribution
        ax1 = axes[0, 0]
        for treat in [0, 1]:
            mask = df_sev['STEROIDS'] == treat
            ax1.hist(df_sev[mask]['propensity_score'], alpha=0.6, bins=30,
                    label=f'Steroids={treat}', density=True)
        ax1.set_xlabel('Propensity Score')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Propensity Score Distribution - Severity {severity}')
        ax1.legend()
        
        # 2. Weight distribution
        ax2 = axes[0, 1]
        ax2.hist(weights, bins=50, edgecolor='black')
        ax2.set_xlabel('Weights')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Weight Distribution - Severity {severity}')
        ax2.axvline(weights.mean(), color='red', linestyle='--', 
                   label=f'Mean: {weights.mean():.2f}')
        ax2.legend()
        
        # 3. Covariate balance
        ax3 = axes[1, 0]
        y_pos = np.arange(len(balance_df))
        ax3.scatter(balance_df['smd_unweighted'], y_pos, alpha=0.6, label='Unweighted')
        ax3.scatter(balance_df['smd_weighted'], y_pos, alpha=0.6, label='Weighted')
        ax3.axvline(-0.1, color='red', linestyle='--', alpha=0.5)
        ax3.axvline(0.1, color='red', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Standardized Mean Difference')
        ax3.set_ylabel('Covariate Index')
        ax3.set_title(f'Covariate Balance - Severity {severity}')
        ax3.legend()
        
        # 4. Outcome by treatment
        ax4 = axes[1, 1]
        outcome_data = df_sev.groupby('STEROIDS')['DEATH_28D'].agg(['mean', 'sem'])
        treatments = ['No Steroids', 'Steroids']
        means = outcome_data['mean'].values
        sems = outcome_data['sem'].values
        x_pos = np.arange(len(treatments))
        ax4.bar(x_pos, means, yerr=sems, capsize=10)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(treatments)
        ax4.set_ylabel('28-Day Mortality Rate')
        ax4.set_title(f'Observed Mortality by Treatment - Severity {severity}')
        
        plt.tight_layout()
        plt.savefig(f'diagnostics_severity_{severity}.png', dpi=300)
        plt.close()
    
    def run_analysis(self, filepath):
        """Run complete causal inference analysis."""
        # Load data
        df = self.load_and_preprocess_data(filepath)
        
        # Analyze by severity level
        for severity in [1, 2, 3]:
            print(f"\n{'='*60}")
            print(f"SEVERITY LEVEL {severity}")
            print(f"{'='*60}")
            
            # Check sample size
            n_sev = len(df[df['SEVERITY_NUMERIC'] == severity])
            if n_sev < 50:
                print(f"Warning: Small sample size (n={n_sev})")
                continue
            
            # Check positivity
            self.check_positivity(df, severity)
            
            # Run different estimation methods
            print("\n1. IPTW Estimation:")
            iptw_res = self.iptw_estimation(df, severity)
            print(f"   Risk (Steroids): {iptw_res['risk_1']:.3f}")
            print(f"   Risk (No Steroids): {iptw_res['risk_0']:.3f}")
            print(f"   ATE (RD): {iptw_res['ate_rd']:.3f}")
            print(f"   ATE (RR): {iptw_res['ate_rr']:.3f}")
            
            print("\n2. G-Computation Estimation:")
            gcomp_res = self.g_computation(df, severity)
            print(f"   Risk (Steroids): {gcomp_res['risk_1']:.3f}")
            print(f"   Risk (No Steroids): {gcomp_res['risk_0']:.3f}")
            print(f"   ATE (RD): {gcomp_res['ate_rd']:.3f}")
            print(f"   ATE (RR): {gcomp_res['ate_rr']:.3f}")
            
            print("\n3. AIPW (Doubly Robust) Estimation:")
            aipw_res = self.aipw_estimation(df, severity)
            print(f"   Risk (Steroids): {aipw_res['risk_1']:.3f}")
            print(f"   Risk (No Steroids): {aipw_res['risk_0']:.3f}")
            print(f"   ATE (RD): {aipw_res['ate_rd']:.3f}")
            print(f"   ATE (RR): {aipw_res['ate_rr']:.3f}")
            
            # Bootstrap CIs for AIPW (primary method)
            print("\n4. Bootstrap Confidence Intervals (AIPW):")
            ci_res = self.bootstrap_confidence_intervals(df, severity, method='aipw', n_bootstrap=500)
            print(f"   95% CI (RD): [{ci_res['ci_rd'][0]:.3f}, {ci_res['ci_rd'][1]:.3f}]")
            print(f"   95% CI (RR): [{ci_res['ci_rr'][0]:.3f}, {ci_res['ci_rr'][1]:.3f}]")
            
            # Check balance
            balance_df = self.check_balance(iptw_res['df_sev'], iptw_res['weights'])
            
            # Create diagnostics
            self.plot_diagnostics(iptw_res['df_sev'], iptw_res['weights'], 
                                severity, balance_df)
            
            # Store results
            self.results[f'severity_{severity}'] = {
                'ate_rd': aipw_res['ate_rd'],
                'ate_rr': aipw_res['ate_rr'],
                'ci_rd_lower': ci_res['ci_rd'][0],
                'ci_rd_upper': ci_res['ci_rd'][1],
                'ci_rr_lower': ci_res['ci_rr'][0],
                'ci_rr_upper': ci_res['ci_rr'][1],
                'rr_estimates': ci_res['rr_estimates'],
                'n': n_sev,
                'n_treated': len(iptw_res['df_sev'][iptw_res['df_sev']['STEROIDS'] == 1]),
                'n_control': len(iptw_res['df_sev'][iptw_res['df_sev']['STEROIDS'] == 0])
            }
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save results to CSV."""
        results_list = []
        
        for severity in [1, 2, 3]:
            if f'severity_{severity}' in self.results:
                res = self.results[f'severity_{severity}']
                results_list.append({
                    'severity': severity,
                    'ate_rd': res['ate_rd'],
                    'ci_rd_lower': res['ci_rd_lower'],
                    'ci_rd_upper': res['ci_rd_upper'],
                    'ate_rr': res['ate_rr'],
                    'ci_rr_lower': res['ci_rr_lower'],
                    'ci_rr_upper': res['ci_rr_upper'],
                    'n_total': res['n'],
                    'n_treated': res['n_treated'],
                    'n_control': res['n_control'],
                    'rr_estimates': res['rr_estimates']
                })
        
        results_df = pd.DataFrame(results_list)
        results_df.to_csv('causal_inference_results.csv', index=False)
        print("\nResults saved to 'causal_inference_results.csv'")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY OF RESULTS")
        print("="*60)
        print(results_df.to_string(index=False))


def main():
    """Main execution function."""
    print("Causal Inference Analysis: Steroids Effect on COVID-19 Mortality")
    print("="*60)
    
    # Initialize analysis
    analysis = CausalInferenceAnalysis()
    
    # Run analysis
    # Note: Replace 'data.csv' with actual filename
    analysis.run_analysis('sdy1662.csv')
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
