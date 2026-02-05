"""Тесты для модуля features."""

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from fraudguard.features import add_basic_features, build_preprocessor


class TestAddBasicFeatures:
    """Тесты для add_basic_features."""

    def test_returns_copy(self):
        """Функция не должна модифицировать исходный DataFrame."""
        df = pd.DataFrame({
            "step": [1, 2, 3],
            "type": ["PAYMENT", "CASH_OUT", "TRANSFER"],
            "amount": [100.0, 250.5, 300.0],
        })

        df2 = add_basic_features(df)

        assert df is not df2
        assert list(df.columns) == ["step", "type", "amount"]

    def test_adds_time_features(self):
        """При наличии transaction_time должны добавляться hour и dayofweek."""
        df = pd.DataFrame({
            "amount": [100.0],
            "transaction_time": ["2025-01-15 14:30:00"],  # Среда
        })

        result = add_basic_features(df)

        assert "hour" in result.columns
        assert "dayofweek" in result.columns
        assert result["hour"].iloc[0] == 14
        assert result["dayofweek"].iloc[0] == 2  # Среда

    def test_handles_missing_transaction_time(self):
        """Без transaction_time функция должна работать без ошибок."""
        df = pd.DataFrame({
            "amount": [100.0, 200.0],
            "type": ["PAYMENT", "TRANSFER"],
        })

        result = add_basic_features(df)

        assert "hour" not in result.columns
        assert "dayofweek" not in result.columns


class TestBuildPreprocessor:
    """Тесты для build_preprocessor."""

    @pytest.fixture
    def sample_df(self):
        """Создаёт тестовый DataFrame."""
        return pd.DataFrame({
            "step": [1, 2, 3],
            "type": ["PAYMENT", "CASH_OUT", "TRANSFER"],
            "amount": [100.0, 250.5, 300.0],
            "nameOrig": ["C123", "C456", "C789"],
        })

    def test_returns_column_transformer(self, sample_df):
        """Должен возвращать ColumnTransformer."""
        preprocessor, _, _ = build_preprocessor(sample_df)
        assert isinstance(preprocessor, ColumnTransformer)

    def test_separates_numeric_and_categorical(self, sample_df):
        """Числовые и категориальные колонки не должны пересекаться."""
        _, num_cols, cat_cols = build_preprocessor(sample_df)

        assert set(num_cols).isdisjoint(set(cat_cols))
        assert "amount" in num_cols
        assert "step" in num_cols
        assert "type" in cat_cols
        assert "nameOrig" in cat_cols

    def test_excludes_target_columns(self):
        """Таргет не должен попадать в признаки."""
        df = pd.DataFrame({
            "amount": [100.0, 200.0],
            "type": ["PAYMENT", "TRANSFER"],
            "isFraud": [0, 1],
            "is_fraud": [0, 1],
        })

        _, num_cols, cat_cols = build_preprocessor(df)

        all_features = num_cols + cat_cols
        assert "isFraud" not in all_features
        assert "is_fraud" not in all_features

    def test_preprocessor_can_fit_transform(self, sample_df):
        """Препроцессор должен успешно трансформировать данные."""
        preprocessor, _, _ = build_preprocessor(sample_df)

        result = preprocessor.fit_transform(sample_df)

        assert result is not None
        assert result.shape[0] == len(sample_df)
