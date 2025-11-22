import numpy as np
import pandas as pd

from fraudguard.features import build_preprocessor
from fraudguard.models import build_logreg_model, build_forest_model


def make_dummy_dataset():
    df = pd.DataFrame(
        {
            "step": [1, 2, 3, 4, 5, 6],
            "type": ["PAYMENT", "CASH_OUT", "PAYMENT", "TRANSFER", "PAYMENT", "CASH_OUT"],
            "amount": [100.0, 5000.0, 50.0, 7000.0, 200.0, 4500.0],
            "nameOrig": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "nameDest": ["M1", "M2", "M3", "M4", "M5", "M6"],
            "oldbalanceOrg": [1000, 6000, 500, 10000, 1500, 8000],
            "newbalanceOrg": [900, 1000, 450, 3000, 1300, 4000],
            "oldbalanceDest": [0, 0, 0, 0, 0, 0],
            "newbalanceDest": [100, 5000, 50, 7000, 200, 4500],
            "isFraud": [0, 1, 0, 1, 0, 0],
        }
    )
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]
    return X, y


def _fit_and_check_model(model, X, y):
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(y)
    # только 0 и 1
    assert set(np.unique(preds)).issubset({0, 1})


def test_logreg_model_trains_and_predicts():
    X, y = make_dummy_dataset()
    preprocessor, _, _ = build_preprocessor(X)

    model = build_logreg_model(preprocessor)
    _fit_and_check_model(model, X, y)


def test_forest_model_trains_and_predicts():
    X, y = make_dummy_dataset()
    preprocessor, _, _ = build_preprocessor(X)

    model = build_forest_model(preprocessor)
    _fit_and_check_model(model, X, y)
