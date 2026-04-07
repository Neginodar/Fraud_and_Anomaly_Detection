# Fraud and Anomaly Detection — Credit Card Transactions

Production-style implementation of supervised and unsupervised fraud detection on the Kaggle credit card dataset, with time-based evaluation, cost-sensitive thresholding, and an inference API.

## Project Overview

This project compares **supervised fraud classification** with **unsupervised anomaly detection** on highly imbalanced credit card transaction data. The pipeline is organized as a Python package with separate modules for data preprocessing, modelling, evaluation, visualization, and deployment.

**Key ideas:**

- Use a **time-based split** (past → train, future → test) to avoid data leakage from random shuffling.  
- Train and compare a **Logistic Regression baseline** and a **LightGBM classifier**, tuning LightGBM with `GridSearchCV` on a `TimeSeriesSplit` using **PR-AUC** as the objective.[file:61][file:56]  
- Benchmark **unsupervised anomaly detectors** (Isolation Forest, One-Class SVM, KMeans) as label-free baselines.[file:55]  
- Optimize the decision **threshold via a cost function** that weights false negatives more heavily than false positives.  
- Expose a **FastAPI** endpoint for real-time fraud scoring.

## Repository Structure

- `src/config.py` — Global configuration (paths, seeds, thresholds).  
- `src/data_preprocessing.py` — Data loading, time-based split, preprocessing pipeline.  
- `src/models_supervised.py` — Logistic Regression and LightGBM definitions + class weights.  
- `src/models_unsupervised.py` — Isolation Forest, One-Class SVM, KMeans, cluster→anomaly helper.  
- `src/evaluation.py` — Metrics (precision, recall, F1, ROC-AUC, PR-AUC) and cost-based threshold search.  
- `src/train_supervised.py` — Train/evaluate supervised models, run `GridSearchCV` for LightGBM, log metrics, and save pipelines.[file:61]  
- `src/train_unsupervised.py` — Train/evaluate unsupervised models and log their metrics.  
- `src/results_helper.py` — Helpers for writing metrics CSVs and managing the `results/` directory.  
- `src/viz.py` — ROC and Precision–Recall curve plotting utilities.  
- `src/api.py` — FastAPI service exposing a `/score` endpoint for online scoring.

Generated artifacts:

- `models/` — Serialized pipelines (model + preprocessing + threshold).  
- `results/supervised_metrics.csv` — Metrics for supervised models (LogReg, LightGBM).[file:56]  
- `results/unsupervised_metrics.csv` — Metrics for unsupervised models (IF, OCSVM, KMeans).[file:55]  
- `results/plots/roc_*.png`, `results/plots/pr_*.png` — ROC and PR curves for supervised models (logreg, lgbm).[file:57][file:58][file:59][file:60]

## Quickstart

1. **Download the dataset**

   Download the Kaggle *Credit Card Fraud Detection* dataset and place it at:

   ```text
   data/creditcard.csv
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   # Fraud and Anomaly Detection — Credit Card Transactions

   This repository contains a production-style implementation comparing supervised fraud classification and unsupervised anomaly detection on the Kaggle Credit Card Fraud dataset.

   Dataset
   -------

   The experiments use the Kaggle *Credit Card Fraud Detection* dataset:

   - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

   Place the downloaded CSV at `data/creditcard.csv` before running training.

   Why this project
   ----------------

   - Demonstrates time-based splitting (past → train, future → test) to avoid leakage.  
   - Compares simple baselines (Logistic Regression) with a tuned LightGBM classifier.  
   - Benchmarks unsupervised anomaly detectors (Isolation Forest, One-Class SVM, KMeans) as label-free baselines.  
   - Uses a cost-sensitive threshold search to pick operating points that prioritize catching frauds over reducing false alarms.  
   - Provides a minimal `FastAPI` inference endpoint for online scoring.

   Repository layout
   -----------------

   - `src/` — Python package with core modules: preprocessing, training, evaluation, viz, and API.  
      - `src/config.py` — global configuration (paths, seeds, cost weights).  
      - `src/data_preprocessing.py` — loading, cleaning, preprocessing pipelines, and time-based split.  
      - `src/models_supervised.py` — supervised model pipelines (LogReg, LightGBM).  
      - `src/models_unsupervised.py` — unsupervised detectors and helpers.  
      - `src/train_supervised.py` — script to train and evaluate supervised models.  
      - `src/train_unsupervised.py` — script to train/evaluate unsupervised detectors.  
      - `src/evaluation.py` — metrics, PR/ROC utilities, and cost-based threshold search.  
      - `src/api.py` — FastAPI app exposing a `/score` endpoint for single-transaction scoring.  
   - `data/` — place `creditcard.csv` here (not committed).  
   - `models/` — saved pipelines after training.  
   - `results/` — CSV metrics and plots produced by training scripts.  

   Quickstart
   ----------

   1. Create and activate a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   2. Download the dataset from the Kaggle link above and save it as `data/creditcard.csv`.

   3. Train supervised models (Logistic Regression + LightGBM):

   ```bash
   python -m src.train_supervised
   ```

   4. (Optional) Train unsupervised detectors:

   ```bash
   python -m src.train_unsupervised
   ```

   5. Run the API locally for online scoring:

   ```bash
   uvicorn src.api:app --reload
   ```

   API example
   -----------

   POST a single transaction to `/score` (JSON body with feature vector). Example payload and `curl`:

   ```bash
   curl -sS -X POST "http://localhost:8000/score" \
      -H "Content-Type: application/json" \
      -d '{"features": {"V1": -1.3598071336738, "V2": -0.0727811733098497, "V3": 2.53634673796914, "Amount": 149.62}}'
   ```

   The response includes a `score` (probability or anomaly score) and a `label` (binary decision using the stored pipeline threshold).

   Results
   -------

   - Trained pipelines are saved to `models/`.  
   - Supervised metrics (ROC-AUC, PR-AUC, precision, recall, expected cost) are written to `results/supervised_metrics.csv`.  
   - Unsupervised metrics are written to `results/unsupervised_metrics.csv`.  
   - Plots (ROC / PR curves) are placed under `results/plots/`.

   Notes & extensions
   ------------------

   - Consider adding SHAP explanations for LightGBM feature importances.  
   - Add a `LICENSE` file if you want to choose a specific open-source license (MIT recommended for simple projects).  
   - Add CI (tests) to ensure training scripts still run after changes.

   Acknowledgements & citation
   ---------------------------

   Original dataset: M. Dal Pozzolo, O. Caelen, R.A. Johnson and G. Bontempi — "Credit Card Fraud Detection" (Kaggle). See the Kaggle dataset page above for citation details.

   License
   -------

   This repository does not include the dataset. Check the Kaggle dataset page for its terms. Code in this repo is available under the MIT License (add a `LICENSE` file to declare formally).

   ----

   If you'd like, I can also:

   - add a short `examples/` notebook demonstrating end-to-end training and scoring, or  
   - generate a minimal `LICENSE` file (MIT) and a `CONTRIBUTING.md` template.
