# Fraud and Anomaly Detection — Credit Card Transactions

Production-style implementation of supervised and unsupervised fraud detection on the Kaggle Credit Card Fraud Detection dataset, with time-based evaluation, cost-sensitive thresholding, and an inference API.

## Project Overview

This project compares **supervised fraud classification** with **unsupervised anomaly detection** on highly imbalanced credit card transaction data. The pipeline is organized as a Python package with separate modules for data preprocessing, modelling, evaluation, visualization, and deployment.

**Key ideas:**

- Use a **time-based split** (past → train, future → test) to avoid data leakage from random shuffling.
- Train and compare a **Logistic Regression baseline** and a **LightGBM classifier**, tuning LightGBM with `GridSearchCV` on a `TimeSeriesSplit` using **PR-AUC** as the objective.
- Benchmark **unsupervised anomaly detectors** (Isolation Forest, One-Class SVM, KMeans) as label-free baselines.
- Optimize the decision **threshold via a cost function** that weights false negatives more heavily than false positives.
- Expose a **FastAPI** endpoint for real-time fraud scoring.

## Dataset

The experiments use the Kaggle **Credit Card Fraud Detection** dataset:

- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download the dataset and place the CSV at:

```text
data/creditcard.csv
```

This file is **not** tracked in the repository.

## Repository Structure

- `src/` — Python package with core modules: preprocessing, training, evaluation, visualization, and API.
  - `src/config.py` — Global configuration (paths, seeds, cost weights, thresholds).
  - `src/data_preprocessing.py` — Data loading, cleaning, preprocessing pipelines, and time-based split.
  - `src/models_supervised.py` — Logistic Regression and LightGBM model pipelines + class weights.
  - `src/models_unsupervised.py` — Unsupervised detectors (Isolation Forest, One-Class SVM, KMeans) and cluster→anomaly helpers.
  - `src/train_supervised.py` — Train and evaluate supervised models, run `GridSearchCV` for LightGBM, log metrics, and save pipelines.
  - `src/train_unsupervised.py` — Train and evaluate unsupervised models and log their metrics.
  - `src/evaluation.py` — Metrics (precision, recall, F1, ROC-AUC, PR-AUC) and cost-based threshold search.
  - `src/results_helper.py` — Helpers for writing metrics CSVs and managing the `results/` directory.
  - `src/viz.py` — ROC and Precision–Recall curve plotting utilities.
  - `src/api.py` — FastAPI app exposing a `/score` endpoint for single-transaction scoring.
- `data/` — Place `creditcard.csv` here (ignored by Git).
- `models/` — Serialized pipelines (model + preprocessing + threshold) saved after training.
- `results/` — CSV metrics and plots produced by training scripts:
  - `results/supervised_metrics.csv` — Metrics for supervised models (LogReg, LightGBM).
  - `results/unsupervised_metrics.csv` — Metrics for unsupervised models (IF, OCSVM, KMeans).
  - `results/plots/roc_*.png`, `results/plots/pr_*.png` — ROC and PR curves for supervised models.

## Quickstart

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download the dataset**

   Download from Kaggle and save as:

   ```text
   data/creditcard.csv
   ```

3. **Train supervised models (Logistic Regression + LightGBM)**

   ```bash
   python -m src.train_supervised
   ```

4. **(Optional) Train unsupervised detectors**

   ```bash
   python -m src.train_unsupervised
   ```

5. **Run the API locally for online scoring**

   ```bash
   uvicorn src.api:app --reload
   ```

## API Example

POST a single transaction to `/score` (JSON body with feature vector). Example:

```bash
curl -sS -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{"features": {"V1": -1.3598071336738, "V2": -0.0727811733098497, "V3": 2.53634673796914, "Amount": 149.62}}'
```

The response includes a `score` (probability or anomaly score) and a `label` (binary decision using the stored pipeline threshold).

## Results

- Trained pipelines are saved to `models/`.
- Supervised metrics (ROC-AUC, PR-AUC, precision, recall, expected cost) are written to `results/supervised_metrics.csv`.
- Unsupervised metrics are written to `results/unsupervised_metrics.csv`.
- Plots (ROC / PR curves) are placed under `results/plots/`.

## Notes & Extensions

- Add SHAP explanations for LightGBM feature importances.
- Add a `LICENSE` file (MIT recommended) to formally specify the code license.
- Add CI (tests) to ensure training scripts still run after changes.
- Add an `examples/` notebook showing end-to-end training and scoring.

## Acknowledgements & License

Original dataset: “Credit Card Fraud Detection” by M. Dal Pozzolo, O. Caelen, R.A. Johnson and G. Bontempi (see the Kaggle dataset page for citation details).

This repository does **not** include the dataset. Check the Kaggle dataset page for its terms. Code in this repo is intended to be MIT-licensed; add a `LICENSE` file to declare it formally.
