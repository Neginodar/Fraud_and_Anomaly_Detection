
# import os
# import joblib
# from src.config import DATA_PATH, MODEL_DIR, RANDOM_STATE
# from src.data_preprocessing import (
#     load_data,
#     train_test_time_split,
#     build_preprocess_pipeline,
#     prepare_features,
# )
# from src.models_supervised import get_logreg_model, get_lgbm_model, compute_class_weights
# from src.evaluation import evaluate_classification, find_best_threshold
# def main():
#     os.makedirs(MODEL_DIR, exist_ok=True)
#     df = load_data(DATA_PATH)
#     train_df, test_df = train_test_time_split(df)
#     X_train, y_train = prepare_features(train_df)
#     X_test, y_test = prepare_features(test_df)

#     preproc = build_preprocess_pipeline()
#     X_train_t = preproc.fit_transform(X_train)
#     X_test_t = preproc.transform(X_test)

#     class_weight = compute_class_weights(y_train)

#     models = {
#         'logreg': get_logreg_model(class_weight=class_weight),
#         'lgbm': get_lgbm_model(class_weight=class_weight)
#     }

#     for name, model in models.items():
#         model.fit(X_train_t, y_train)
#         if hasattr(model, 'predict_proba'):
#             y_score = model.predict_proba(X_test_t)[:, 1]
#         else:
#             y_score = model.decision_function(X_test_t)
#         thr, _ = find_best_threshold(y_test.values, y_score)
#         y_pred = (y_score >= thr).astype(int)
#         metrics = evaluate_classification(y_test.values, y_pred, y_score)
#         print(f"Model: {name}, threshold: {thr:.3f}, metrics: {metrics}")

#         joblib.dump({'model': model, 'preproc': preproc, 'threshold': thr}, os.path.join(MODEL_DIR, f'{name}_pipeline.joblib'))

# if __name__ == '__main__':
#     main()
import os
import joblib

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from lightgbm import LGBMClassifier

from src.config import DATA_PATH, MODEL_DIR, RANDOM_STATE
from src.data_preprocessing import (
    load_data,
    train_test_time_split,
    build_preprocess_pipeline,
    prepare_features,
)
from src.models_supervised import (
    get_logreg_model,
    compute_class_weights,
)
from src.evaluation import evaluate_classification, find_best_threshold
from src.results_helper import RESULTS_DIR, append_metrics_csv
from src.viz import plot_roc, plot_pr


SUP_METRICS_CSV = os.path.join(RESULTS_DIR, "supervised_metrics.csv")


def get_tuned_lgbm_model(class_weight=None):
    """LightGBM with GridSearchCV on a time-series split, optimized for PR-AUC."""
    base_model = LGBMClassifier(
        class_weight=class_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    param_grid = {
        "n_estimators": [200, 500],
        "learning_rate": [0.01, 0.05],
        "num_leaves": [31, 63],
        "min_child_samples": [20, 50],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
    }

    tscv = TimeSeriesSplit(n_splits=3)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="average_precision",  # PR-AUC
        cv=tscv,
        n_jobs=-1,
        verbose=1,
    )

    return grid


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    train_df, test_df = train_test_time_split(df)
    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    preproc = build_preprocess_pipeline()
    X_train_t = preproc.fit_transform(X_train)
    X_test_t = preproc.transform(X_test)

    class_weight = compute_class_weights(y_train)

    models = {
        "logreg": get_logreg_model(class_weight=class_weight),
        "lgbm": get_tuned_lgbm_model(class_weight=class_weight),
    }

    header = [
        "model",
        "threshold",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "expected_cost",
    ]

    for name, model in models.items():
        # For LightGBM, model is a GridSearchCV; for LogReg, it's the estimator
        model.fit(X_train_t, y_train)

        if isinstance(model, GridSearchCV):
            print(f"Best params for {name}: {model.best_params_}")
            estimator = model.best_estimator_
        else:
            estimator = model

        if hasattr(estimator, "predict_proba"):
            y_score = estimator.predict_proba(X_test_t)[:, 1]
        else:
            y_score = estimator.decision_function(X_test_t)

        thr, _ = find_best_threshold(y_test.values, y_score)
        y_pred = (y_score >= thr).astype(int)
        metrics = evaluate_classification(y_test.values, y_pred, y_score)
        print(f"Model: {name}, threshold: {thr:.3f}, metrics: {metrics}")

        # Save pipeline
        joblib.dump(
            {"model": estimator, "preproc": preproc, "threshold": thr},
            os.path.join(MODEL_DIR, f"{name}_pipeline.joblib"),
        )

        # Log metrics
        row = [
            name,
            thr,
            metrics.get("precision"),
            metrics.get("recall"),
            metrics.get("f1"),
            metrics.get("roc_auc"),
            metrics.get("pr_auc"),
            metrics.get("expected_cost"),
        ]
        append_metrics_csv(SUP_METRICS_CSV, header, row)

        # Plots
        if metrics.get("roc_auc") is not None:
            plot_roc(y_test.values, y_score, name)
        if metrics.get("pr_auc") is not None:
            plot_pr(y_test.values, y_score, name)


if __name__ == "__main__":
    main()