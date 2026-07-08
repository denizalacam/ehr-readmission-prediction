"""
DeepEHR Baseline Model

Train and evaluate a Logistic Regression baseline model.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)

from data_pipeline import (
    load_dataset,
    create_X_y,
    split_data,
    create_processed_feature_matrix,
)


RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
    )

    metrics = {
        "Accuracy": accuracy,
        "ROC-AUC": auc,
        "Precision": report_dict["1"]["precision"],
        "Recall": report_dict["1"]["recall"],
        "F1-Score": report_dict["1"]["f1-score"],
    }

    print("\nModel Evaluation")
    print("-" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    return metrics


def plot_roc_curve(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_feature_importance(model, feature_names):
    coefficients = model.coef_[0]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    )

    importance_df = importance_df.sort_values(
        by="absolute_coefficient",
        ascending=False,
    )

    output_path = RESULTS_DIR / "feature_importance.csv"
    importance_df.to_csv(output_path, index=False)

    print("\nTop 10 Most Important Features")
    print(importance_df.head(10))
    print(f"\nSaved feature importance to: {output_path}")


def save_confusion_matrix_plot(model, X_test, y_test):
    y_pred = model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Not Readmitted", "Readmitted"],
    )

    plt.title("Confusion Matrix - Logistic Regression")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nSaved confusion matrix plot.")


def save_metrics_report(metrics):
    output_path = RESULTS_DIR / "metrics.txt"

    with open(output_path, "w") as f:
        f.write("DeepEHR Baseline Logistic Regression Results\n")
        f.write("=" * 50 + "\n\n")

        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

    print(f"\nSaved metrics report to: {output_path}")


def save_model(model, preprocessor):
    model_path = MODELS_DIR / "logistic_regression.pkl"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved preprocessor to: {preprocessor_path}")


def main():
    dataset = load_dataset()

    X, y = create_X_y(dataset)

    X_train, X_test, y_train, y_test = split_data(X, y)

    (
        X_train_processed,
        X_test_processed,
        preprocessor,
    ) = create_processed_feature_matrix(
        X_train,
        X_test,
    )

    print("\nOriginal Training Shape")
    print(X_train.shape)

    print("\nProcessed Training Shape")
    print(X_train_processed.shape)

    model = train_logistic_regression(
        X_train_processed,
        y_train,
    )

    metrics = evaluate_model(
        model,
        X_test_processed,
        y_test,
    )

    plot_roc_curve(
        model,
        X_test_processed,
        y_test,
    )

    save_feature_importance(
        model,
        X_train_processed.columns,
    )

    save_confusion_matrix_plot(
        model,
        X_test_processed,
        y_test,
    )

    save_metrics_report(metrics)

    save_model(
        model,
        preprocessor,
    )


if __name__ == "__main__":
    main()