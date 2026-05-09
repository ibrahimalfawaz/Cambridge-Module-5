# Predicting 30-Day Hospital Readmission in Diabetic Patients Using Machine Learning
### A Comparative Study of Logistic Regression and Random Forest

**Author:** Ibrahim Al Fawaz  
**Programme:** MSt Healthcare Data Science — University of Cambridge  
**Module:** Module 5 — Machine Learning in Healthcare  

---

## Abstract

**Background:** Hospital readmission within 30 days is a globally recognised clinical quality metric used by the US Centers for Medicare and Medicaid Services to penalise institutions with higher-than-expected rates. Diabetic patients face elevated readmission risk due to disease complexity and comorbidity burden, making this population a priority target for predictive modelling.

**Objective:** To develop and compare machine learning models for predicting 30-day readmission risk in hospitalised diabetic patients, with a focus on clinically justified preprocessing, systematic hyperparameter tuning, and explicit comparison of class imbalance correction strategies.

**Methods:** The publicly available Diabetes 130-US Hospitals dataset (101,766 encounters, 1999–2008) was used. A clinically grounded preprocessing pipeline was implemented: terminal discharge records were excluded, missing HbA1c values were encoded as "Not_tested" rather than discarded, ICD-9 codes were grouped into ten clinical categories, and a polypharmacy score was computed. Feature selection combined variance thresholding and Chi-squared testing, reducing 192 candidate features to 72. Data were split into training (72%), validation (8%), and test (20%) sets with stratification. Logistic Regression and Random Forest were trained with class weighting; Random Forest hyperparameters were tuned on the validation set. Undersampling and SMOTE were additionally evaluated.

**Results:** The untuned Random Forest severely overfitted (Train AUC = 1.000, Validation AUC = 0.646, Recall = 0.01). Hyperparameter tuning (max_depth=6, min_samples_leaf=10) resolved overfitting (gap = 0.013) and increased recall to 0.60. The tuned Random Forest outperformed Logistic Regression on recall (0.60 vs. 0.54) and ROC-AUC (0.668 vs. 0.664). SMOTE and undersampling were inferior to tuning. Prior inpatient visits, discharge destination, and prior emergency visits were the strongest predictors.

**Conclusion:** Hyperparameter tuning proved more impactful than resampling for managing class imbalance in this dataset. The tuned Random Forest provides clinically meaningful identification of high-risk patients. Ethical considerations including fairness monitoring across racial groups and clinician-led threshold selection are discussed.

---

## Dataset

| Item | Detail |
|---|---|
| **Name** | Diabetes 130-US Hospitals for Years 1999–2008 |
| **Source** | UCI Machine Learning Repository |
| **Size** | 101,766 encounters, 50 features |
| **Direct CSV link** | https://raw.githubusercontent.com/ibrahimalfawaz/Cambridge-Module-5/main/diabetic_data.csv |
| **Original paper** | Strack et al. (2014), BioMed Research International, DOI: 10.1155/2014/781670 |

> The dataset is loaded automatically in the notebook via the raw GitHub URL — no manual download is required.

---

## Repository Structure
Cambridge-Module-5/
│
├── README.md                               <- This file
├── diabetic_data.csv                       <- Dataset (loaded automatically)
└── HDS_ML_IbrahimAlFawaz.ipynb             <- Main Colab analysis notebook

---

## How to Reproduce the Results

### Option 1 — Google Colab (Recommended, no setup required)

1. Click the notebook file `HDS_ML_IbrahimAlFawaz.ipynb` in this repository
2. Click **"Open in Colab"** at the top of the file
3. In Colab, go to **Runtime > Run All**
4. All data loads automatically from the GitHub raw URL — no manual upload needed

> The notebook is fully self-contained. Running all cells top to bottom reproduces all results, tables, and figures in the report.

### Option 2 — Local Python Environment

```bash
# 1. Clone the repository
git clone https://github.com/ibrahimalfawaz/Cambridge-Module-5.git
cd Cambridge-Module-5

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook HDS_ML_IbrahimAlFawaz.ipynb
```

---

## Library Versions

All analysis was conducted in **Google Colab** using **Python 3.12.3**.

| Library | Version | Purpose |
|---|---|---|
| `pandas` | 3.0.2 | Data loading, cleaning, manipulation |
| `numpy` | 2.4.4 | Numerical operations and arrays |
| `matplotlib` | 3.10.8 | Visualisation and plots |
| `scikit-learn` | 1.8.0 | ML models, feature selection, evaluation |
| `imbalanced-learn` | 0.12.4 | SMOTE and undersampling |
| `scipy` | 1.17.1 | Statistical testing |
| `Python` | 3.12.3 | Base language (Google Colab default) |

### requirements.txt
pandas==3.0.2
numpy==2.4.4
matplotlib==3.10.8
scikit-learn==1.8.0
imbalanced-learn==0.12.4
scipy==1.17.1

> In Google Colab, all libraries except `imbalanced-learn` are pre-installed.  
> The first notebook cell installs it automatically:
> ```python
> !pip install imbalanced-learn==0.12.4
> ```

---

## Pipeline Overview

| Step | Description |
|---|---|
| 1 | Load data from GitHub raw URL |
| 2 | Replace `"?"` with NaN, remove terminal discharge records |
| 3 | Missing data handling (drop, impute, encode as "Not_tested") |
| 4a | ICD-9 grouping, medication encoding, polypharmacy score |
| 4b | Admission/discharge/source code one-hot encoding |
| 5 | Variance threshold + Chi-squared feature selection (192 → 72 features) |
| 6 | Column renaming for clinical interpretability |
| 7 | Train / Validation / Test split (72/8/20, stratified) + StandardScaler |
| 8 | Logistic Regression with class_weight="balanced" (baseline) |
| 9 | Random Forest: hyperparameter tuning on validation set |
| 10 | Evaluation: Recall, F1, ROC-AUC, confusion matrix, feature importance, threshold analysis |

---

## Key Results

| Model | Recall | Precision | F1 | ROC-AUC | Overfit Gap |
|---|---|---|---|---|---|
| Naive Baseline | 0.00 | N/A | 0.00 | 0.500 | — |
| Logistic Regression | 0.54 | 0.18 | 0.27 | 0.664 | 0.002 |
| **RF Tuned (Best)** | **0.60** | 0.18 | 0.27 | **0.668** | 0.013 |

> **Primary metric: Recall** — missing a patient who will be readmitted (false negative) is clinically more harmful than unnecessarily flagging a stable patient (false positive).

---

## References

1. Strack et al. (2014). Impact of HbA1c measurement on hospital readmission rates. *BioMed Research International*. https://doi.org/10.1155/2014/781670
2. Emi-Johnson & Nkrumah (2025). Predicting 30-day readmission in diabetes using ML. *Cureus*. https://doi.org/10.7759/cureus.82437
3. Alghatani et al. (2021). 30-day readmission risk in diabetic patients. PMC8323261.
4. CMS. (2023). Hospital Readmissions Reduction Program. https://www.cms.gov/medicare/quality/value-based-programs/hospital-readmissions
5. Probst et al. (2019). Hyperparameters and tuning strategies for Random Forest. *WIREs DMKD*. https://arxiv.org/abs/1804.03515
6. Chen et al. (2024). Clinical validity of SMOTE is questionable. *MAKE*, 6(2), 39. https://doi.org/10.3390/make6020039

> Full reference list available in the written report.

---

*MSt Healthcare Data Science — University of Cambridge — Module 5*
