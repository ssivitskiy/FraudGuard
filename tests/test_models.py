"""Тесты для модуля models."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fraudguard.features import build_preprocessor
from fraudguard.models import build_forest_model, build_logreg_model


@pytest.fixture
def dummy_dataset():
    """Создаёт синтетический датасет для тестов."""
    df = pd.DataFrame(
        {
            "step": [1, 2, 3, 4, 5, 6],
            "type": ["PAYMENT", "CASH_OUT", "PAYMENT", "TRANSFER", "PAYMENT", "CASH_OUT"],
            "amount": [100.0, 5000.0, 50.0, 7000.0, 200.0, 4500.0],
            "nameOrig": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "nameDest": ["M1", "M2", "M3", "M4", "M5", "M6"],
            "oldbalanceOrg": [1000, 6000, 500, 10000, 1500, 8000],
            "newbalanceOrg": [900, 1000, 450, 3000, 1300, 4000],
            "isFraud": [0, 1, 0, 1, 0, 0],
        }
    )
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]
    return X, y


class TestLogRegModel:
    """Тесты для логистической регрессии."""

    def test_build_returns_pipeline(self, dummy_dataset):
        """Должен возвращать Pipeline."""
        X, _ = dummy_dataset
        preprocessor, _, _ = build_preprocessor(X)

        model = build_logreg_model(preprocessor)

        assert isinstance(model, Pipeline)
        assert "preprocess" in model.named_steps
        assert "clf" in model.named_steps

    def test_trains_and_predicts(self, dummy_dataset):
        """Модель должна обучаться и выдавать корректные предсказания."""
        X, y = dummy_dataset
        preprocessor, _, _ = build_preprocessor(X)

        model = build_logreg_model(preprocessor)
        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == len(y)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_returns_valid_probabilities(self, dummy_dataset):
        """Вероятности должны быть в диапазоне [0, 1]."""
        X, y = dummy_dataset
        preprocessor, _, _ = build_preprocessor(X)

        model = build_logreg_model(preprocessor)
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (len(X), 2)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestForestModel:
    """Тесты для Random Forest."""

    def test_build_returns_pipeline(self, dummy_dataset):
        """Должен возвращать Pipeline."""
        X, _ = dummy_dataset
        preprocessor, _, _ = build_preprocessor(X)

        model = build_forest_model(preprocessor)

        assert isinstance(model, Pipeline)
        assert "preprocess" in model.named_steps
        assert "clf" in model.named_steps

    def test_trains_and_predicts(self, dummy_dataset):
        """Модель должна обучаться и выдавать корректные предсказания."""
        X, y = dummy_dataset
        preprocessor, _, _ = build_preprocessor(X)

        model = build_forest_model(preprocessor, n_estimators=10)  # Меньше деревьев для скорости
        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == len(y)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_custom_parameters(self, dummy_dataset):
        """Кастомные параметры должны применяться."""
        X, _ = dummy_dataset
        preprocessor, _, _ = build_preprocessor(X)

        model = build_forest_model(
            preprocessor,
            n_estimators=50,
            max_depth=5,
            min_samples_leaf=2,
        )

        clf = model.named_steps["clf"]
        assert clf.n_estimators == 50
        assert clf.max_depth == 5
        assert clf.min_samples_leaf == 2
