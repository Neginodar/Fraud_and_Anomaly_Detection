
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

def evaluate_classification(y_true, y_pred, y_score=None, cost_fp=1.0, cost_fn=5.0):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    if y_score is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_score)
            metrics['pr_auc'] = average_precision_score(y_true, y_score)
        except Exception:
            metrics['roc_auc'] = np.nan
            metrics['pr_auc'] = np.nan
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    metrics['expected_cost'] = cost_fp * fp + cost_fn * fn
    return metrics

def find_best_threshold(y_true, y_score, cost_fp=1.0, cost_fn=5.0, n_points=50):
    thresholds = np.linspace(0.0, 1.0, n_points)
    best_thr, best_cost = 0.5, float('inf')
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_cost:
            best_cost = cost
            best_thr = thr
    return best_thr, best_cost
