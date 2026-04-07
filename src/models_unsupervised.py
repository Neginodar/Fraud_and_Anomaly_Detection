
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
import numpy as np

def get_isolation_forest():
    return IsolationForest(contamination='auto', n_estimators=200, random_state=42)

def get_oneclass_svm():
    return OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')

def get_kmeans(n_clusters=2):
    return KMeans(n_clusters=n_clusters, random_state=42)

def kmeans_labels_to_anomaly(labels):
    counts = np.bincount(labels)
    minority_label = counts.argmin()
    return (labels == minority_label).astype(int)
