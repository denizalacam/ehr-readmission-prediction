"""
Train and evaluate an XGBoost model for EHR readmission prediction.
"""

from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from data_pipeline import (
    load_dataset,
    create_X_y,
    split_data,
    create_processed_feature_matrix,
)


# ---------------------------------------------------
# Make directories
# ---------------------------------------------------

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def train_xgboost_model(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
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

    print("\nXGBoost Model Evaluation")
    print("-" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    return metrics


def save_xgboost_metrics(metrics):
    output_path = RESULTS_DIR / "xgboost_metrics.txt"

    with open(output_path, "w") as f:
        f.write("EHRPredict XGBoost Results\n")
        f.write("=" * 40 + "\n\n")

        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

    print(f"\nSaved XGBoost metrics to: {output_path}")


def save_xgboost_model(model):
    output_path = MODELS_DIR / "xgboost_model.pkl"

    joblib.dump(model, output_path)

    print(f"\nSaved XGBoost model to: {output_path}")


def save_xgboost_feature_importance(model, feature_names):
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    )

    csv_path = RESULTS_DIR / "xgboost_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    top_features = importance_df.head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(
        top_features["feature"][::-1],
        top_features["importance"][::-1],
    )
    plt.xlabel("Importance")
    plt.title("Top 10 XGBoost Feature Importances")
    plt.tight_layout()

    plot_path = RESULTS_DIR / "xgboost_feature_importance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved XGBoost feature importance CSV to: {csv_path}")
    print(f"Saved XGBoost feature importance plot to: {plot_path}")

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

    model = train_xgboost_model(
        X_train_processed,
        y_train,
    )

    metrics = evaluate_model(
        model,
        X_test_processed,
        y_test,
    )

    save_xgboost_metrics(metrics)

    save_xgboost_model(model)

    save_xgboost_feature_importance(
    model,
    X_train_processed.columns,
)


if __name__ == "__main__":
    main()