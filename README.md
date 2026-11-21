# RabTech Task 04 – Supervised Classification Modeling & Tuning

## Project overview
This project predicts loan default (`loan_default`) using the synthetic credit-risk dataset prepared for the previous feature-engineering task. Four classification algorithms are trained, tuned with `GridSearchCV`, evaluated on a held-out test set, and compared using ROC-AUC and classification metrics.

## Dataset
- File: `credit-default-risk.csv`
- Records: 1,200
- Target: `loan_default` (`0` = no default, `1` = default)
- Features include age, annual income, loan amount, credit score, employment years, debt-to-income ratio, loan term, education, employment type, home ownership, and loan purpose.
- Missing values are handled inside the model pipeline.

**Important:** This is a synthetic educational dataset. This project is not a validated credit-scoring system and must not be used to make real lending decisions.

## Models trained
1. Logistic Regression
2. Random Forest
3. Extra Trees
4. Gradient Boosting

Each model is paired with a preprocessing pipeline:
- Median imputation and standard scaling for numeric columns
- Most-frequent imputation and one-hot encoding for categorical columns

## Validation and evaluation
- 80/20 stratified train-test split
- 3-fold `StratifiedKFold` cross-validation
- `GridSearchCV` hyperparameter tuning
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Visuals: ROC curves and confusion matrices

## Repository structure
```text
rabtech-task04/
├── credit-default-risk.csv
├── supervised_classification_modeling.ipynb
├── README.md
├── implementation_note.md
├── requirements.txt
└── outputs/
    ├── champion_model.joblib
    ├── champion_model_info.txt
    ├── model_comparison_cv.csv
    ├── model_evaluation_test.csv
    ├── roc_auc_curves.png
    └── confusion_matrices.png
```

## Run locally (VS Code / macOS)
1. Open this folder in VS Code.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open `supervised_classification_modeling.ipynb` and select the `.venv` Python kernel.
5. Run all notebook cells from top to bottom.

The notebook expects `credit-default-risk.csv` in the project root and writes evaluation artifacts to `outputs/`.

## Main result
The champion is selected by the highest mean cross-validation ROC-AUC, not by the held-out test score. See `outputs/model_comparison_cv.csv` and `outputs/model_evaluation_test.csv` for the measured results.
