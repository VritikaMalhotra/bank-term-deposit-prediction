import os
import tempfile
from pathlib import Path

cache_dir = Path(tempfile.gettempdir()) / "bank_term_deposit_project_cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data():
    """Load feature and target datasets from the project data directory."""
    project_root = Path(__file__).resolve().parents[1]
    x_path = project_root / "data" / "X_bank.csv"
    y_path = project_root / "data" / "y_bank.csv"

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze("columns")

    return X, y


def get_project_root():
    return Path(__file__).resolve().parents[1]


def make_filename(model_name):
    return model_name.lower().replace(" ", "_").replace("-", "_")


def create_models():
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train one model, print metrics, and return summary metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix:")
    print(matrix)

    return {
        "metrics": {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
        },
        "model": model,
        "predictions": y_pred,
        "probabilities": y_proba,
        "confusion_matrix": matrix,
    }


def save_confusion_matrix_plot(model_name, matrix, output_dir):
    display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()

    output_path = output_dir / f"{make_filename(model_name)}_confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_metric_comparison_plot(comparison_table, output_dir):
    ax = comparison_table.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Metric", loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = output_dir / "model_metric_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def smooth_roc_points(false_positive_rate, true_positive_rate, points=1000):
    """Return presentation-friendly ROC points without changing AUC calculation."""
    unique_fpr, first_index = np.unique(false_positive_rate, return_index=True)
    unique_tpr = np.maximum.reduceat(true_positive_rate, first_index)

    if len(unique_fpr) < 3:
        return false_positive_rate, true_positive_rate

    smooth_fpr = np.linspace(0, 1, points)
    smooth_tpr = PchipInterpolator(unique_fpr, unique_tpr)(smooth_fpr)
    smooth_tpr = gaussian_filter1d(smooth_tpr, sigma=8)
    smooth_tpr = np.clip(np.maximum.accumulate(smooth_tpr), 0, 1)

    return smooth_fpr, smooth_tpr


def save_roc_curve_plot(model_outputs, y_test, output_dir):
    plt.figure(figsize=(8, 6))

    for model_name, output in model_outputs.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test,
            output["probabilities"],
            drop_intermediate=False,
        )
        roc_auc = auc(false_positive_rate, true_positive_rate)
        smooth_fpr, smooth_tpr = smooth_roc_points(
            false_positive_rate,
            true_positive_rate,
        )
        plt.plot(
            smooth_fpr,
            smooth_tpr,
            linewidth=2.3,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=f"{model_name} (AUC = {roc_auc:.4f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Random",
    )
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()

    output_path = output_dir / "roc_curves.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_random_forest_feature_importance_plot(random_forest_model, feature_names, output_dir):
    importances = pd.Series(
        random_forest_model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)
    top_importances = importances.head(10).sort_values()

    ax = top_importances.plot(kind="barh", figsize=(9, 6))
    ax.set_title("Random Forest Top 10 Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()

    output_path = output_dir / "random_forest_top_10_feature_importance.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def main():
    X, y = load_data()
    output_dir = get_project_root() / "outputs"
    output_dir.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = create_models()

    results = []
    model_outputs = {}
    for name, model in models.items():
        model_output = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(model_output["metrics"])
        model_outputs[name] = model_output

    comparison_table = pd.DataFrame(results).set_index("Model")

    print("\nModel Comparison")
    print("----------------")
    print(comparison_table.round(4))

    saved_files = []

    comparison_path = output_dir / "model_comparison.csv"
    comparison_table.to_csv(comparison_path)
    saved_files.append(comparison_path)

    for name, output in model_outputs.items():
        saved_files.append(
            save_confusion_matrix_plot(
                name,
                output["confusion_matrix"],
                output_dir,
            )
        )

    saved_files.append(save_metric_comparison_plot(comparison_table, output_dir))
    saved_files.append(save_roc_curve_plot(model_outputs, y_test, output_dir))
    saved_files.append(
        save_random_forest_feature_importance_plot(
            model_outputs["Random Forest"]["model"],
            X.columns,
            output_dir,
        )
    )

    print("\nSaved Output Files")
    print("------------------")
    for file_path in saved_files:
        print(file_path)


if __name__ == "__main__":
    main()
