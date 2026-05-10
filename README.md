# Bank Term Deposit Classification

This project builds and compares machine learning classification models for a bank marketing term deposit dataset. The goal is to predict whether a customer will subscribe to a term deposit using prepared feature and target CSV files.

It also includes basic data preprocessing and exploratory data analysis (EDA), including class distribution, selected feature relationships, a correlation heatmap, and an outlier boxplot.

## Project Structure

```text
bank_term_deposit_project/
├── data/
│   ├── X_bank.csv
│   ├── y_bank.csv
│   └── bank_cleaned.csv
├── notebooks/
│   └── CMPE255GroupProject.ipynb   # data preprocessing pipeline
├── src/
│   └── modeling.py
├── outputs/
│   └── generated model results and plots
├── slides_notes/
│   ├── Plans.docx
│   └── EDA.docx
├── .gitignore
└── README.md
```

## Models

The modeling script trains and evaluates four classifiers:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curve and AUC

## Outputs

Running the script creates an `outputs/` folder and saves:

- `model_comparison.csv`
- `logistic_regression_confusion_matrix.png`
- `decision_tree_confusion_matrix.png`
- `random_forest_confusion_matrix.png`
- `xgboost_confusion_matrix.png`
- `model_metric_comparison.png`
- `roc_curves.png`
- `random_forest_top_10_feature_importance.png`

The `outputs/` folder is ignored by git because these files are generated artifacts.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
```

macOS / Linux:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install pandas matplotlib scipy scikit-learn xgboost
```

## Run the Project

From the project root, run:

```bash
python src/modeling.py
```

If using the project virtual environment directly:

```bash
.venv/bin/python src/modeling.py
```

The script prints model metrics in the terminal and saves the comparison table and plots to `outputs/`.

## Data

The script expects these files:

- `data/X_bank.csv`: feature matrix
- `data/y_bank.csv`: target labels

The train/test split is stratified, with 80% of the data used for training and 20% used for testing.
