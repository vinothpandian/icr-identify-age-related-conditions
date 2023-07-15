import numpy as np
from sklearn import metrics


def balanced_log_loss(y_true, y_pred):
    nc = np.bincount(y_true)
    return metrics.log_loss(y_true, y_pred, sample_weight=1 / nc[y_true], eps=1e-15)
