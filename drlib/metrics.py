import numpy as np
from sklearn.metrics import confusion_matrix

def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    O = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes)).astype(float)
    N = O.sum()
    w = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            w[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)
    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist) / N if N else np.zeros_like(O)
    num = (w * O).sum()
    den = (w * E).sum()
    return 1.0 - num / den if den > 0 else 0.0
