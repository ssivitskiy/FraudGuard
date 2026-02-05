from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    confusion_matrix: np.ndarray
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    classification_report: str

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "Model Evaluation Results",
            "=" * 50,
            "",
            "Confusion Matrix:",
            str(self.confusion_matrix),
            "",
            f"Precision: {self.precision:.4f}",
            f"Recall:    {self.recall:.4f}",
            f"F1-Score:  {self.f1:.4f}",
        ]

        if self.roc_auc is not None:
            lines.append(f"ROC-AUC:   {self.roc_auc:.4f}")
        if self.pr_auc is not None:
            lines.append(f"PR-AUC:    {self.pr_auc:.4f}")

        lines.extend(["", "Classification Report:", self.classification_report])

        return "\n".join(lines)


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y_true: pd.Series,
    threshold: float = 0.5,
) -> EvaluationResult:
    y_pred = model.predict(X)

    y_proba = _get_probabilities(model, X)

    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = None
    pr_auc = None

    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
            pr_auc = average_precision_score(y_true, y_proba)
        except ValueError as e:
            logger.warning("Could not compute AUC metrics: %s", e)

    report = classification_report(y_true, y_pred, digits=4)

    result = EvaluationResult(
        confusion_matrix=cm,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        classification_report=report,
    )

    logger.info("Evaluation complete - F1: %.4f, ROC-AUC: %s", f1, roc_auc)
    return result


def _get_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

    logger.warning("Model has no predict_proba or decision_function")
    return None