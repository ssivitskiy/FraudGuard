import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "transaction_time" in df.columns:
        dt = pd.to_datetime(df["transaction_time"])
        df["hour"] = dt.dt.hour
        df["dayofweek"] = dt.dt.dayofweek

    return df


def build_preprocessor(df: pd.DataFrame):
    df = add_basic_features(df)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in ["is_fraud", "target", "label", "isFraud"]:
        if col in numeric_cols:
            numeric_cols.remove(col)
        if col in categorical_cols:
            categorical_cols.remove(col)

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols
