from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from typing import Tuple

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_raw_data(fname: str = "transactions.csv") -> pd.DataFrame:
    path = DATA_DIR / "raw" / fname
    logger.info("Loading data from %s", path)

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. "
            "Please download the dataset and place it in data/raw/"
        )

    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def train_valid_test_split(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
    test_size: float = 0.2,
    valid_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info("Target distribution: %s", y.value_counts().to_dict())

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    valid_ratio = valid_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=valid_ratio, stratify=y_temp, random_state=random_state
    )

    logger.info(
        "Split sizes - Train: %d, Valid: %d, Test: %d",
        len(X_train),
        len(X_valid),
        len(X_test),
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test