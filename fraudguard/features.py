from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

TARGET_COLUMNS = frozenset({"is_fraud", "isFraud", "target", "label", "fraud"})


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "transaction_time" in df.columns:
        dt = pd.to_datetime(df["transaction_time"], errors="coerce")
        df["hour"] = dt.dt.hour
        df["dayofweek"] = dt.dt.dayofweek
        logger.debug("Added time features: hour, dayofweek")

    return df


def build_preprocessor(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    df = add_basic_features(df)

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_cols = [col for col in numeric_cols if col.lower() not in TARGET_COLUMNS]
    categorical_cols = [col for col in categorical_cols if col.lower() not in TARGET_COLUMNS]

    if "transaction_time" in categorical_cols:
        categorical_cols.remove("transaction_time")

    logger.info("Numeric features (%d): %s", len(numeric_cols), numeric_cols)
    logger.info("Categorical features (%d): %s", len(categorical_cols), categorical_cols)

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor, numeric_cols, categorical_cols
