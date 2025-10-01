Midwest Healthcare Conference Potential Outcomes Challenge: Estimating the Causal Effect of Glucocorticoids on In-Hospital Survival Among COVID-19 Patients
====================================================================================

As part of the 4th Midwest Healthcare Conference we invite you to participate in a mini-data challenge focused on **estimating the causal effect of systemic glucocorticoids ("steroids") on in-hospital survival for patients with COVID-19**.  **This track is grounded in the Potential Outcomes / Neyman-Rubin causal framework**.  Your task is to leverage modern propensity-score and outcome-regression methodology to recover the average treatment effect that would be observed under random assignment of steroids, using the real-world dataset provided.

--------------------------------------------------------------------
Challenge Overview
--------------------------------------------------------------------
Observational data rarely satisfy the conditions of a randomized controlled trial (RCT).  The Potential Outcomes framework provides a rigorous statistical language for reasoning about *counterfactual* outcomes and for constructing estimators that, under clearly stated assumptions, recover causal effects from such data.  Participants will submit an analytic plan and code that estimate the causal **risk difference and risk ratio** in 28-day in-hospital mortality between two potential worlds stratified by baseline severity level (SEVERITY_NUMERIC: 1,2,3):

* Y(1) – the patient receives systemic glucocorticoids within 24 h of admission
* Y(0) – the patient does **not** receive systemic glucocorticoids within 24 h of admission

The primary estimand is the **Average Treatment Effect (ATE)** in the stratified population of interest.

ATE = E[ Y(1) − Y(0) ].

Challenge: Utilize Potential Outcomes to reproduce high-quality randomized controlled trial findings on the treatment of Covid-19 with
steroids using real world data in the stratified severity population of interest.
• Therapeutic Area: Severe Covid-19
• Motivation: Real world data is often significantly biased. This challenge tests the hypothesis that potential outcomes provide a transparent, rigorous approach to reduce biasing correlations when estimating causal effects from observational data.

• Potential outcomes mathematically specify a counterfactual between treatment and outcome.
• They allow for the identification of causal relationships, prediction of outcomes under hypothetical interventions, and determination of the impact of changes within a system.
• Adjustment of variables can introduce or minimize confounding.

--------------------------------------------------------------------
Data Source
--------------------------------------------------------------------
Dataset: ImmPort accession **SDY1662 – “An Inflammatory Cytokine Signature Predicts Covid-19 Severity And Survival.”**  The organisers supply a pre-processed *person-day* format.  You may only use variables contained in this release.

We created a dataset for Steroids trial emulations.
• This allows for focused analysis of the treatment while maintaining the structure needed for our pooled logistic regression approach (specifically addressing eligibility and treatment exposure censoring issues, see below).


We transformed the original dataset into a Person-Time dataset. In a Person-Time dataset, each day of a patient's hospitalization is represented and eligibility criteria, action/treatment exposure, outcomes, and variables are tracked. For a Pooled Logistic Regression analysis, each Hospital Day creates a new cohort and eligibility criteria are applied. Patients are tracked for up to 28 Study Days or censored if 28 Study Days are not available or the patient deviates from the emulated randomization.


• We generated data for up to 28 days from emulated randomization (Study Day 0), where available.
• Missing data was handled using the last observation carried forward method.
  o This approach allows us to maximize the use of available data, however there are several limitations with this strategy.

--------------------------------------------------------------------
Task Requirements
--------------------------------------------------------------------
1. **Causal Assumptions** – Explicitly state the assumptions under which your estimator is unbiased:
   • Consistency
   • Positivity
   • Conditional Exchangeability (Ignorability) given a covariate set **C**.

2. **Identification Strategy** – Describe how ATE is expressed as a functional of the observed data distribution (e.g. g-computation formula, IPTW, AIPW, TMLE).

3. **Estimation Procedure** – Implement one or more of the following:
   • Propensity-score matching / weighting
   • Outcome regression / g-computation
   • Doubly-robust methods (AIPW, TMLE)
   • SuperLearner or other ensemble learners are permitted for nuisance function estimation.

4. **Variance & Uncertainty** – Provide 95 % compatibility (confidence) intervals via non-parametric bootstrap (≥ 500 draws) or influence-function-based standard errors.

5. **Diagnostics** – Include diagnostics that demonstrate adequacy of overlap and covariate balance (e.g. propensity-score density plots, standardized mean differences before/after weighting).

--------------------------------------------------------------------
Deliverables
--------------------------------------------------------------------
1. **Results file** (CSV or JSON) containing
   • Point estimate of ATE risk difference and risk ratio for each severity level
   • 95 % lower & upper bounds for each severity level
2. **Methodological report** (≤ 3 pages PDF) detailing
   • Covariates used in C
   • Identification and estimation approach
   • Diagnostic figures
   • Discussion of limitations
3. **Reproducible code** (Jupyter notebook or script) runnable with `python3 -m pip install -r requirements.txt && python analysis.py`.

--------------------------------------------------------------------
Evaluation Metrics
--------------------------------------------------------------------
Primary metric
• **Coverage & proximity to RCT benchmark** – Deviation of your ATE estimate and 95 % interval from seminal steroid RCT results (e.g. RECOVERY-Dexamethasone).

Secondary metrics
• Covariate balance after adjustment for each severity level
• Soundness of causal assumptions & clarity of documentation
• Computational efficiency and code reproducibility

--------------------------------------------------------------------
Ethical & Privacy Considerations
--------------------------------------------------------------------
All data are de-identified.  Participation requires acceptance of the ImmPort User Agreement (https://docs.immport.org/home/agreement/).  Share only aggregate results in your write-up.

--------------------------------------------------------------------
Important Dates
--------------------------------------------------------------------
• Data release: **Sept 20, 2025**
• Submission deadline: **Oct 25, 2025, 23:59 PT**
• Winners announced: **Nov 5, 2025** at the Midwest Healthcare Conference closing session.

--------------------------------------------------------------------
Getting Started
--------------------------------------------------------------------
We provide a data dictionary to get you started.

Good luck, and happy causal inference!
