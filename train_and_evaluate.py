import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from feature_extraction import FEATURE_COLUMNS

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def validate_dataset(df: pd.DataFrame) -> None:
    required = FEATURE_COLUMNS + ["Label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def build_models() -> dict:
    return {
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=3)),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(C=10.0, kernel="rbf", gamma="scale")),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }


def save_visual_reports(df: pd.DataFrame, y_test, y_pred, label_encoder: LabelEncoder, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Class distribution over the whole dataset.
    class_counts = df["Label"].value_counts().sort_index()
    plt.figure(figsize=(24, 8))
    plt.bar(class_counts.index, class_counts.values, color="#3b82f6")
    plt.title("Class Distribution (Samples per Person)")
    plt.xlabel("Person Label")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=180)
    plt.close()

    # Confusion matrix on the holdout split.
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    labels = sorted(set(y_test_labels) | set(y_pred_labels))

    fig, ax = plt.subplots(figsize=(20, 18))
    ConfusionMatrixDisplay.from_predictions(
        y_test_labels,
        y_pred_labels,
        labels=labels,
        normalize="true",
        xticks_rotation=90,
        colorbar=True,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Normalized Confusion Matrix (Holdout Set)")
    ax.tick_params(axis="both", which="major", labelsize=5)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Finger Geometry classifiers")
    parser.add_argument("--data", default="all_features.csv", help="Path to feature CSV")
    parser.add_argument("--model-out", default="models/finger_geometry_model.joblib", help="Output path for saved model")
    parser.add_argument("--report-out", default="models/model_report.json", help="Output path for metrics report")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    validate_dataset(df)

    X = df[FEATURE_COLUMNS].copy()
    y_raw = df["Label"].astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}

    models = build_models()
    cv_results = {}

    print("Running 5-fold stratified cross-validation...")
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[name] = {
            "accuracy_mean": float(scores["test_accuracy"].mean()),
            "accuracy_std": float(scores["test_accuracy"].std()),
            "f1_macro_mean": float(scores["test_f1_macro"].mean()),
            "f1_macro_std": float(scores["test_f1_macro"].std()),
        }

    ranking = sorted(cv_results.items(), key=lambda item: item[1]["f1_macro_mean"], reverse=True)

    print("\nModel ranking (by CV macro-F1):")
    for rank, (name, metrics) in enumerate(ranking, start=1):
        print(
            f"{rank}. {name:<14} "
            f"acc={metrics['accuracy_mean']:.4f}±{metrics['accuracy_std']:.4f} "
            f"f1_macro={metrics['f1_macro_mean']:.4f}±{metrics['f1_macro_std']:.4f}"
        )

    best_name = ranking[0][0]
    best_model = models[best_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    holdout = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "test_size": int(len(y_test)),
    }

    holdout_classification_report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    artifact = {
        "model": best_model,
        "label_encoder": label_encoder,
        "feature_columns": FEATURE_COLUMNS,
        "best_model_name": best_name,
    }

    model_out = Path(args.model_out)
    report_out = Path(args.report_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, model_out)

    report = {
        "dataset_rows": int(len(df)),
        "dataset_classes": int(y_raw.nunique()),
        "cv_results": cv_results,
        "best_model": best_name,
        "holdout": holdout,
    }

    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (report_out.parent / "classification_report.txt").write_text(holdout_classification_report, encoding="utf-8")
    save_visual_reports(df, y_test, y_pred, label_encoder, report_out.parent)

    print("\nBest model:", best_name)
    print("Holdout accuracy:", f"{holdout['accuracy']:.4f}")
    print("Holdout macro-F1:", f"{holdout['f1_macro']:.4f}")
    print(f"Saved model to {model_out}")
    print(f"Saved report to {report_out}")
    print(f"Saved class distribution plot to {report_out.parent / 'class_distribution.png'}")
    print(f"Saved confusion matrix plot to {report_out.parent / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
