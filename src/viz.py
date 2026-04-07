
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

from src.results_helper import PLOTS_DIR


def plot_roc(y_true, y_score, model_name: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve - {model_name}")
    plt.legend()
    out_path = os.path.join(PLOTS_DIR, f"roc_{model_name}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_pr(y_true, y_score, model_name: str):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f"{model_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curve - {model_name}")
    plt.legend()
    out_path = os.path.join(PLOTS_DIR, f"pr_{model_name}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
