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
   source .venv/bin/activate        # macOS / Linux
   # .venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Train supervised models**

   ```bash
   python -m src.train_supervised
   ```

   This will:

   - Train Logistic Regression and a tuned LightGBM model on the time-based training split.  
   - Choose the decision threshold that minimizes expected cost.  
   - Save pipelines under `models/`.  
   - Log metrics to `results/supervised_metrics.csv` and generate ROC/PR plots under `results/plots/`.[file:56][file:57][file:58][file:59][file:60]

5. **Train unsupervised anomaly detectors (optional)**

   ```bash
   python -m src.train_unsupervised
   ```

   This trains Isolation Forest, One-Class SVM, and KMeans on non-fraud data (where applicable), then evaluates them on the test split and logs metrics to `results/unsupervised_metrics.csv`.[file:55]

6. **Run the FastAPI service**

   ```bash
   uvicorn src.api:app --reload
   ```

   The API exposes:

   - `POST /score` — accepts a JSON body with a `features` dictionary (one transaction) and returns a fraud **score** and binary **label** computed using the saved LightGBM pipeline.[file:61]

## Results

### Supervised models (time-based split)

From `results/supervised_metrics.csv`:[file:56]

- **Logistic Regression**  
  - ROC-AUC ≈ 0.986  
  - PR-AUC ≈ 0.76  
  - Precision ≈ 0.51, Recall ≈ 0.81  
  - Expected cost ≈ 128 (with asymmetric FP/FN costs)

- **LightGBM (tuned with GridSearchCV)**  
  - ROC-AUC ≈ 0.986  
  - PR-AUC ≈ 0.81  
  - Precision ≈ 0.92, Recall ≈ 0.76, F1 ≈ 0.83  
  - Expected cost ≈ 95

ROC and PR curves for both models are saved under `results/plots/` and show consistently strong separation between fraud and non-fraud classes.[file:57][file:58][file:59][file:60]

### Unsupervised anomaly detection

From `results/unsupervised_metrics.csv`:[file:55]

- **Isolation Forest**  
  - ROC-AUC ≈ 0.94, PR-AUC ≈ 0.03  
  - High recall but very low precision → expected cost ≈ 2319.

- **One-Class SVM**  
  - ROC-AUC ≈ 0.94, PR-AUC ≈ 0.15  
  - Slightly better precision, but still costly → expected cost ≈ 1043.

- **KMeans (k=2)**  
  - Fails to separate frauds in this setting → expected cost ≈ 786.

This highlights that while unsupervised anomaly detection can surface suspicious transactions without labels, a supervised LightGBM model offers a far superior precision–recall–cost trade-off once labels are available.

## Possible Extensions

- Add SHAP-based feature importance for LightGBM.  
- Perform threshold sweeps to visualize cost vs decision threshold.  
- Experiment with additional models (e.g., XGBoost, CatBoost) or sequence/graph-based fraud representations.