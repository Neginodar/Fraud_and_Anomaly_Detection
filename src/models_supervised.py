
import numpy as np
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight

def compute_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

def get_logreg_model(class_weight=None):
    return LogisticRegression(max_iter=1000, class_weight=class_weight)

def get_lgbm_model(class_weight=None):
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight=class_weight,
    )
