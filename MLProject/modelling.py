import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)

# =============================================
# ARGUMENT PARSER
# =============================================
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators",      type=int,   default=100)
parser.add_argument("--max_depth",         type=int,   default=5)
parser.add_argument("--min_samples_split", type=int,   default=2)
parser.add_argument("--min_samples_leaf",  type=int,   default=1)
parser.add_argument("--train_path",        type=str,   default="iris_preprocessing/iris_train.csv")
parser.add_argument("--test_path",         type=str,   default="iris_preprocessing/iris_test.csv")
args = parser.parse_args()


# =============================================
# LOAD DATA
# =============================================
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    feature_cols = ['sepal length (cm)', 'sepal width (cm)',
                    'petal length (cm)', 'petal width (cm)']
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test  = test_df[feature_cols]
    y_test  = test_df['target']
    return X_train, X_test, y_train, y_test, feature_cols


# =============================================
# ARTEFAK: Confusion Matrix
# =============================================
def save_confusion_matrix(y_test, y_pred, labels, path="confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


# =============================================
# ARTEFAK: Feature Importance
# =============================================
def save_feature_importance(model, feature_cols, path="feature_importance.png"):
    importances = model.feature_importances_
    plt.figure(figsize=(7, 4))
    sns.barplot(x=importances, y=feature_cols, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


# =============================================
# MAIN
# =============================================
if __name__ == "__main__":
    mlflow.set_experiment("Iris-CI-Pipeline")

    X_train, X_test, y_train, y_test, feature_cols = load_data(args.train_path, args.test_path)
    labels = ['setosa', 'versicolor', 'virginica']

    # Log parameters
    mlflow.log_param("n_estimators",      args.n_estimators)
    mlflow.log_param("max_depth",         args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("min_samples_leaf",  args.min_samples_leaf)

    # Train model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Metrics
    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')
    roc_auc   = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')

    mlflow.log_metric("accuracy",  accuracy)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("roc_auc",   roc_auc)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Artefak tambahan
    cm_path = save_confusion_matrix(y_test, y_pred, labels)
    mlflow.log_artifact(cm_path, artifact_path="plots")
    os.remove(cm_path)

    fi_path = save_feature_importance(model, feature_cols)
    mlflow.log_artifact(fi_path, artifact_path="plots")
    os.remove(fi_path)

    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    report_path = "classification_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact(report_path, artifact_path="reports")
    os.remove(report_path)

    print("Training selesai. Artefak disimpan di MLflow.")