import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_raw_data(fname: str = "transactions.csv") -> pd.DataFrame:
    path = DATA_DIR / "raw" / fname
    df = pd.read_csv(path)
    return df

def train_valid_test_split(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
    test_size: float = 0.2,
    valid_size: float = 0.25,
    random_state: int = 42,
):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    valid_ratio = valid_size / (1 - test_size)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=valid_ratio, stratify=y_temp, random_state=random_state
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test
