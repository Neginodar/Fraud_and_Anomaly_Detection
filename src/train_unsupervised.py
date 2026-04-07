
import os
import joblib

from src.config import DATA_PATH, MODEL_DIR
from src.data_preprocessing import (
    load_data,
    train_test_time_split,
    build_preprocess_pipeline,
    prepare_features,
)
from src.models_unsupervised import (
    get_isolation_forest,
    get_oneclass_svm,
    get_kmeans,
    kmeans_labels_to_anomaly,
)
from src.evaluation import evaluate_classification
from src.results_helper import RESULTS_DIR, append_metrics_csv


UNSUP_METRICS_CSV = os.path.join(RESULTS_DIR, "unsupervised_metrics.csv")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    train_df, test_df = train_test_time_split(df)
    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    preproc = build_preprocess_pipeline()
    X_train_t = preproc.fit_transform(X_train)
    X_test_t = preproc.transform(X_test)

    models = {
        "iforest": get_isolation_forest(),
        "ocsvm": get_oneclass_svm(),
        "kmeans": get_kmeans(),
    }

    header = [
        "model",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "expected_cost",
    ]

    for name, model in models.items():
        if name == "kmeans":
            model.fit(X_train_t)
            labels = model.predict(X_test_t)
            y_pred = kmeans_labels_to_anomaly(labels)
            y_score = None
        else:
            model.fit(X_train_t[y_train == 0])
            scores = model.decision_function(X_test_t)
            # Higher score = more anomalous
            y_score = -scores
            thr = 0.0
            y_pred = (y_score >= thr).astype(int)

        metrics = evaluate_classification(y_test.values, y_pred, y_score)
        print(f"Unsupervised model: {name}, metrics: {metrics}")

        # Save pipeline
        joblib.dump(
            {"model": model, "preproc": preproc},
            os.path.join(MODEL_DIR, f"{name}_unsup_pipeline.joblib"),
        )

        # Log metrics
        row = [
            name,
            metrics.get("precision"),
            metrics.get("recall"),
            metrics.get("f1"),
            metrics.get("roc_auc"),
            metrics.get("pr_auc"),
            metrics.get("expected_cost"),
        ]
        append_metrics_csv(UNSUP_METRICS_CSV, header, row)


if __name__ == "__main__":
    main()