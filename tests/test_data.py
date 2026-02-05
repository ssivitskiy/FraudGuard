"""Тесты для модуля data."""

import pandas as pd
import pytest

from fraudguard.data import train_valid_test_split


class TestTrainValidTestSplit:
    """Тесты для train_valid_test_split."""

    @pytest.fixture
    def sample_df(self):
        """Создаёт тестовый DataFrame."""
        return pd.DataFrame(
            {
                "feature1": range(100),
                "feature2": range(100, 200),
                "is_fraud": [0] * 90 + [1] * 10,  # 10% fraud
            }
        )

    def test_returns_six_objects(self, sample_df):
        """Должен возвращать 6 объектов."""
        result = train_valid_test_split(sample_df, target_col="is_fraud")
        assert len(result) == 6

    def test_correct_split_sizes(self, sample_df):
        X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(
            sample_df,
            target_col="is_fraud",
            test_size=0.2,
            valid_size=0.25,
        )
        total = len(sample_df)
        # Test: 20%, Valid: 25% от оставшихся 80% = 20%, Train: 60%
        assert len(X_test) == pytest.approx(total * 0.2, abs=2)
        assert len(X_valid) == pytest.approx(total * 0.8 * 0.25, abs=3)  # Исправлено
        assert len(X_train) == pytest.approx(total * 0.8 * 0.75, abs=3)  # Исправлено

    def test_target_not_in_features(self, sample_df):
        """Таргет не должен быть в признаках."""
        X_train, X_valid, X_test, _, _, _ = train_valid_test_split(sample_df, target_col="is_fraud")

        assert "is_fraud" not in X_train.columns
        assert "is_fraud" not in X_valid.columns
        assert "is_fraud" not in X_test.columns

    def test_stratification_preserved(self, sample_df):
        """Соотношение классов должно сохраняться в выборках."""
        _, _, _, y_train, y_valid, y_test = train_valid_test_split(sample_df, target_col="is_fraud")

        original_ratio = sample_df["is_fraud"].mean()

        # Допускаем небольшое отклонение из-за округления
        assert y_train.mean() == pytest.approx(original_ratio, abs=0.05)
        assert y_valid.mean() == pytest.approx(original_ratio, abs=0.1)
        assert y_test.mean() == pytest.approx(original_ratio, abs=0.1)

    def test_raises_on_missing_target(self, sample_df):
        """Должен выбрасывать KeyError при отсутствии таргета."""
        with pytest.raises(KeyError, match="not found"):
            train_valid_test_split(sample_df, target_col="nonexistent")

    def test_reproducibility(self, sample_df):
        """С одинаковым random_state должны получаться одинаковые разбиения."""
        result1 = train_valid_test_split(sample_df, target_col="is_fraud", random_state=42)
        result2 = train_valid_test_split(sample_df, target_col="is_fraud", random_state=42)

        X_train1, _, _, _, _, _ = result1
        X_train2, _, _, _, _, _ = result2

        pd.testing.assert_frame_equal(X_train1, X_train2)
