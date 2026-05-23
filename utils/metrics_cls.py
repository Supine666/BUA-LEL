# utils/metrics_cls.py
import numpy as np

def compute_classwise_sens_spec(cm: np.ndarray, eps: float = 1e-12):
    C = cm.shape[0]
    sens = np.zeros(C, dtype=np.float64)
    spec = np.zeros(C, dtype=np.float64)

    total = cm.sum()
    for c in range(C):
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = total - TP - FN - FP

        sens[c] = (TP + eps) / (TP + FN + eps)
        spec[c] = (TN + eps) / (TN + FP + eps)

    return sens, spec