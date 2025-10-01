# Supplementary Methods & Materials for "Leveraging Generative AI for Interpretable Clinical Decision Making Through Causal Graphs"

## Methods

### Study Design

We organized the UIUC Midwest Healthcare Conference Challenge to evaluate performance of target trial emulation by participants without formal causal inference training. The challenge used a collaborative user interface in which participants designed structural causal models to replicate the effects of COVID-19 treatment of the RECOVERY trial using observational data. The RECOVERY Trial results were used as a standard against which to benchmark participant entries due to its randomized controlled trial design -- the gold standard for clinical studies when implementation is feasible. One year later, we conducted a comparative analysis using four state-of-the-art generative AI models (OpenAI o3, DeepSeek R1, Gemini Pro 2.5, Claude Opus) for causal diagram construction. Conference session documentation is available in `prompt_materials/PSBsession.md`.

### Participants and Data Access

Participants were recruited through conference announcements without background restrictions. They received comprehensive metadata for the SDY1662 dataset including complete data dictionaries (`prompt_materials/sdy1662_data_dictionary.csv`), variable definitions, and statistical profiles, but no direct patient-level data access. All analyses occurred via secure API calls via the cStructure interface.

### Dataset Characteristics

We utilized the SDY1662 dataset from ImmPort containing >2,000 hospitalized COVID-19 patients from Mount Sinai Health System (March-May 2020). The dataset was transformed into person-time format for pooled logistic regression, creating ~65,000 person-day observations. Complete dataset specifications are documented in `prompt_materials/sdy1662_data_dictionary.csv`.

Disease severity was classified using a three-level ordinal scale based on oxygen support requirements: mild-moderate disease (room air or low-flow oxygen, SpO₂ ≥ 94% on ≤ 4L O₂), severe disease (high-flow oxygen, non-invasive ventilation, or mechanical ventilation without vasopressors), and critical disease (mechanical ventilation with vasopressor support or evidence of end-organ dysfunction defined as creatinine clearance < 30 mL/min or alanine aminotransferase > 5× upper limit of normal). Missing data was handled using last observation carried forward, with variables >50% missing excluded from analysis. The data supporting this publication is available at ImmPort (immport.org) under study accession SDY1662 An Inflammatory Cytokine Signature Predicts Covid-19 Severity And Survival.

#### Pooled Mini-Cohorts

To minimize immortal time bias, we created daily mini-cohorts evaluating treatment-eligible patients and assigning them to treatment strategies reflecting observed exposure. Patients could appear multiple times under different assignments, with censoring upon treatment strategy deviation. For example, a patient with their first steroid exposure on hospital day 2 would be a 'never treat' patient in mini-cohort 0 and 1, but a '10 days of steroids' patient in mini-cohort 2.

### Target Trial Emulation Framework

Participants designed models to estimate causal effects of glucocorticoids and hydroxychloroquine on 28-day all-cause mortality in hospitalized COVID-19 patients, stratified by disease severity. We applied RECOVERY trial eligibility criteria: hospitalized patients ≥18 years with COVID-19, no treatment contraindications, and sufficient follow-up data. Treatment comparisons evaluated glucocorticoid versus standard care, and hydroxychloroquine versus standard care (results focus on glucocorticoid treatment).

### Platform and Technical Implementation

#### cStructure Platform

The platform implements a browser-based collaborative environment featuring: interactive DAG editor with real-time collaborative editing, automated d-separation analysis providing feedback on confounding pathways, and JSON serialization of models transmitted via REST API to remote analysis servers. Patient data remained on secure servers while participants constructed models using metadata only. Real-time d-separation algorithms (implemented in TypeScript) identified open backdoor paths between treatment and outcome nodes, providing automated feedback on confounding control adequacy. Platform documentation is available in `prompt_materials/scm/`.

### Model Construction

#### Human-generated SCMs

Participants constructed directed acyclic graphs specifying: nodes from available dataset features, directional edges representing causal relationships, scientific rationale documentation, and node classifications (Action/Outcome/Adjusted/Unadjusted). Automated validation included d-separation analysis, temporal ordering checks, and convergence assessment. Final human-generated models are stored in `scms/graphml/`, `scms/json/`, and visualized in `scms/png/`.

#### Transformer-generated SCMs

Each AI model received similar instructions and materials provided to human participants, including the UIUC challenge announcement, SDY1662 data dictionary, cStructure platform description, and GraphML causal graph specification format (available in `prompt_materials/scm/`). We developed a standardized prompt requesting causal diagrams in GraphML format for evaluating steroid safety and efficacy on 28-day survival in hospitalized COVID-19 patients. AI-generated structural causal models are available in `scms/graphml/` (GraphML format), `scms/json/` (JSON specifications), and `scms/png/` (visual representations).

#### SCM Ablation using Potential Outcomes

To test whether causal graphs provide unique benefits beyond LLM reasoning capabilities, we conducted a controlled ablation experiment using multiple state-of-the-art reasoning models (OpenAI o3, Claude Opus, Gemini Pro 2.5, and DeepSeekV1) on identical causal inference tasks. Each model was presented with a Potential Outcomes task that requested the use of traditional causal inference methods (IPTW, AIPW, g-computation) without causal graph structure. The potential outcomes task used identical data dictionaries, bootstrap requirements (n=500), and a similar description of the target trial emulation. The experimental design isolated the effect of structured causal reasoning by holding constant the underlying statistical objectives, data access, model capabilities, and evaluation framework while varying only the presence of explicit causal graph scaffolding. Prompt materials for the potential outcomes tasks are available in `prompt_materials/potential_outcomes/`, with corresponding analysis scripts organized by model in `potential_outcomes/claude/`, `potential_outcomes/deepseekr1/`, `potential_outcomes/gemini/`, and `potential_outcomes/openai/`.

### TTE Statistical Analysis Framework

Causal effects were estimated using the parametric g-formula (PyGFormula v1.1.6) with 500 bootstrap samples per disease severity stratum (random seed 123). This estimator was utilized to overcome statistical convergence issues encountered by participants of the Midwest Healthcare Conference Challenge. Bootstrap sampling occurred at the patient level to preserve within-patient correlations. Risk ratios compared predicted 28-day survival probabilities under treatment versus standard of care. Model-specific analysis implementations are available in `potential_outcomes/` subdirectories, with each AI model's scripts including bootstrap metrics computation and results processing.

### Model Comparison Evaluation Metrics

#### Causal Plausibility Assessment

To evaluate the scientific validity of proposed causal relationships, three independent clinical reviewers (blinded to model source) assessed all directed edges from human-generated (n=36 edges) and Transformer-generated (n=86 edges) structural causal models using a standardized ordinal scale: 0=no plausible causal pathway exists, 1=weak/speculative rationale with limited mechanistic support, 2=moderate plausibility with established biological mechanisms, 3=strong evidence from randomized trials or robust observational studies. 

**Reviewer training:** Prior to evaluation, reviewers received standardized instructions defining each scale point with clinical examples (available in `prompt_materials/causal_plausibility/`), emphasizing assessment of biological plausibility rather than statistical associations, with each edge evaluated independently without knowledge of the complete model structure or authorship. 

**Statistical analysis:** We employed a Bayesian hierarchical model with crossed random effects accounting for between-model variation (models nested within groups) and reviewer measurement error (reviewers crossed, evaluating all edges), using the specification `Score ~ Group + (1|Model_ID) + (1|Reviewer_ID)` with weakly informative priors (Group effect: Normal(0, 0.5); random effect SDs: HalfNormal(1)). The model was implemented in Python using Bambi/PyMC with 2,000 posterior draws across 4 chains (target_accept=0.95), with convergence assessed via R̂ (<1.01) and effective sample size (>400). Plausibility ratings for all causal relationships are documented in `results/causal_edges_plausibility_scores.csv`.

#### Coverage Analysis

Bootstrapped confidence interval coverage against published RECOVERY trial results using 500 bootstrap samples per disease severity stratum. Models were ranked by the total number of bootstrap estimates falling outside published RECOVERY trial 95% confidence intervals across all severity strata (n=1,500 total bootstrap estimates per model). Two-way binomial tests compared each model's coverage performance against the Age base model to assess statistical significance of differences in bootstrap outlier counts. Severity-stratified forest plots showing treatment effect estimates are available in `results/severity1_forestplot.png`, `results/severity2_forestplot.png`, and `results/severity3_forestplot.png`.

#### Sign Error Analysis

Directional accuracy assessment where a sign error was defined as a bootstrap point estimate with the opposite sign of the corresponding severity-specific RECOVERY trial point estimate. Models were ranked by total sign errors across 1,500 bootstrap iterations (500 per severity level) and two-way binomial test p-values were generated using the same procedures as the coverage analysis. Error analysis results are documented in `results/sign_errors.pdf`.

---

Statistical analysis used Python 3.13.3 with two-way binomial test p-values (scipy v1.15.2), adjusted for multiple comparisons (n=7). Plots were generated using Matplotlib and Seaborn python packages.