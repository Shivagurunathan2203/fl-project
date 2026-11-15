from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, accuracy_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_class: int = 5,
    y_pred_backdoor: np.ndarray = None,
) -> dict[str, float]:
    """
    Computes accuracy, macro-F1, macro-precision, and Attack Success Rate (ASR).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))

    asr = 0.0
    if y_pred_backdoor is not None:
        y_pred_backdoor = np.asarray(y_pred_backdoor)
        # Fraction of test samples with true label != target_class that are predicted as target_class
        mask = (y_true != target_class)
        if mask.any():
            asr = float((y_pred_backdoor[mask] == target_class).mean())

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "macro_precision": prec,
        "asr": asr,
    }
