# Model Training and Evaluation — Bharat Horizon Bank

## 1. Purpose
Train a simple, interpretable Logistic Regression classifier to predict the synthetic dataset's `loan_outcome` label. This is a college educational prototype, not a real credit-underwriting model.

## 2. Dataset checked
- File: `bharat_horizon_home_loan_dataset_v2.xlsx`
- Rows: 3,000
- Approved labels: 2,100 (70.0%)
- Rejected labels: 900 (30.0%)
- Train/test split: 2,400 training rows / 600 test rows (80/20, stratified, random seed 42)

The `default_timing_demo_only` column is deliberately excluded from model features because it is demonstration metadata, not an input intended for prediction. The model uses the other 10 listed features, including the two derived ratios.

## 3. Method
- Model: Logistic Regression in a scikit-learn Pipeline.
- Numeric features: median imputation and standardization.
- Categorical feature (`employment_type`): most-frequent imputation and one-hot encoding.
- Target: `loan_outcome` (`Approved` / `Rejected`).
- Approval is treated as the positive class for precision, recall, and F1.
- Baselines: predict every application as Rejected; and predict every application as Approved.

## 4. Test-set results

| Metric | Logistic Regression | Baseline: all Rejected | Baseline: all Approved |
|---|---:|---:|---:|
| Accuracy | 76.2% | 30.0% | 70.0% |
| Approval precision | 80.6% | 0.0% | 70.0% |
| Approval recall | 86.9% | 0.0% | 100.0% |
| Approval F1 | 83.6% | 0.0% | 82.3% |

### Confusion matrix
Rows are actual labels; columns are predictions.

|  | Predicted Rejected | Predicted Approved |
|---|---:|---:|
| Actual Rejected | 92 | 88 |
| Actual Approved | 55 | 365 |

- False approvals (actual Rejected, predicted Approved): **88**
- False rejections (actual Approved, predicted Rejected): **55**

## 5. Interpretation
The Logistic Regression model scores 76.2% accuracy on this held-out test split, compared with 70.0% for the all-Approved majority-class baseline and 30.0% for the all-Rejected baseline. Approval precision is 80.6% and approval recall is 86.9%.

These results are for the supplied synthetic labels only. The dataset was designed to have a 70/30 label split with intentional noisy exceptions, so a model can learn label-generation patterns that do not fully match the written lending policy. Do not interpret these metrics as evidence of real-world creditworthiness, fairness, or bank performance.

## 6. Human checkpoint and policy guardrails
The model output is a recommendation, not a final lending decision. The app should separately check policy criteria and flag cases for human review when policy conditions are borderline, information is missing, the case involves a prior default, or the policy does not cover the scenario. The reviewer must be able to override the recommendation.

## 7. Reproducibility
Run from the folder containing the dataset and `train_model.py`:

```bash
pip install pandas openpyxl scikit-learn joblib
python train_model.py --data "bharat_horizon_home_loan_dataset_v2.xlsx" --output-dir artifacts
```

The script writes a serialized pipeline, JSON metrics, and a Markdown evaluation report to the output folder.
