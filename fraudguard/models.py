from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)


def build_logreg_model(
    preprocessor: ColumnTransformer,
    max_iter: int = 1000,
    C: float = 1.0,
) -> Pipeline:
    clf = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        n_jobs=-1,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    logger.info("Built LogisticRegression model (C=%.2f, max_iter=%d)", C, max_iter)
    return model


def build_forest_model(
    preprocessor: ColumnTransformer,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
) -> Pipeline:
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    logger.info(
        "Built RandomForest model (n_estimators=%d, max_depth=%s)",
        n_estimators,
        max_depth,
    )
    return model