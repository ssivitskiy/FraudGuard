from typing import Any
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


def evaluate_model(model: Any, X, y_true) -> None:
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = model.decision_function(X)
        y_proba = (y_proba - np.min(y_proba)) / (np.max(y_proba) - np.min(y_proba))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))

    try:
        roc = roc_auc_score(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        print(f"ROC-AUC: {roc:.4f}")
        print(f"PR-AUC:  {pr_auc:.4f}")
    except Exception as e:
        print("ROC/PR calculation error:", e)
