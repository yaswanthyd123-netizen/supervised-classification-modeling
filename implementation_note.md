# Implementation Note – Task 04

## Objective
Build and compare supervised classification models for predicting the `loan_default` target in the credit-default-risk dataset.

## Implementation
1. Loaded the 1,200-row CSV dataset and inspected its columns, target distribution, and missing values.
2. Separated the input features from the binary target.
3. Split the data into 80% training and 20% testing sets using stratification.
4. Built a `ColumnTransformer` preprocessing pipeline:
   - Numeric columns: median imputation followed by standard scaling.
   - Categorical columns: most-frequent imputation followed by one-hot encoding.
5. Trained and tuned four classifiers:
   - Logistic Regression
   - Random Forest
   - Extra Trees
   - Gradient Boosting
6. Used `GridSearchCV` with 3-fold `StratifiedKFold` and ROC-AUC scoring.
7. Evaluated each tuned estimator on the held-out test set using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
8. Generated ROC-AUC curves and confusion matrices.
9. Selected the champion using the highest cross-validation ROC-AUC and serialized the full fitted preprocessing/model pipeline as `outputs/champion_model.joblib`.

## Results
The highest mean cross-validation ROC-AUC was achieved by **Logistic Regression (0.6669)**. Its held-out test ROC-AUC was **0.6318**. The test-set precision was **0.3333**, recall was **0.4844**, and F1-score was **0.3949**.

These values are included as observed experimental results, not as evidence of a production-ready credit model. See the CSV result files for the complete model comparison.

## Reproducibility
- Random seed: 42
- Test size: 20%
- Cross-validation: 3-fold stratified
- Tuning: GridSearchCV
- Champion artifact: Joblib

## Limitations and responsible use
The dataset is synthetic and contains class imbalance. The observed metrics are modest and should be interpreted cautiously. No fairness, explainability, external validation, or real-world deployment assessment has been completed. Do not use this model for actual lending, eligibility, or credit decisions.
