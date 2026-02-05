"""Тесты для модуля evaluate."""

import numpy as np
import pandas as pd

from fraudguard.evaluate import EvaluationResult, evaluate_model


class MockModel:
    """Мок-модель для тестов."""

    def __init__(self, predictions, probabilities):
        self._predictions = predictions
        self._probabilities = probabilities

    def predict(self, X):
        return self._predictions

    def predict_proba(self, X):
        return np.column_stack([1 - self._probabilities, self._probabilities])


class TestEvaluateModel:
    """Тесты для evaluate_model."""

    def test_returns_evaluation_result(self):
        """Должен возвращать EvaluationResult."""
        y_true = pd.Series([0, 0, 1, 1])
        model = MockModel(
            predictions=np.array([0, 0, 1, 1]),
            probabilities=np.array([0.1, 0.2, 0.8, 0.9]),
        )

        result = evaluate_model(model, pd.DataFrame({"x": [1, 2, 3, 4]}), y_true)

        assert isinstance(result, EvaluationResult)

    def test_perfect_predictions(self):
        """При идеальных предсказаниях метрики должны быть равны 1."""
        y_true = pd.Series([0, 0, 1, 1])
        model = MockModel(
            predictions=np.array([0, 0, 1, 1]),
            probabilities=np.array([0.0, 0.0, 1.0, 1.0]),
        )

        result = evaluate_model(model, pd.DataFrame({"x": [1, 2, 3, 4]}), y_true)

        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.roc_auc == 1.0

    def test_confusion_matrix_shape(self):
        """Confusion matrix должна быть 2x2."""
        y_true = pd.Series([0, 0, 1, 1])
        model = MockModel(
            predictions=np.array([0, 1, 0, 1]),
            probabilities=np.array([0.2, 0.6, 0.4, 0.8]),
        )

        result = evaluate_model(model, pd.DataFrame({"x": [1, 2, 3, 4]}), y_true)

        assert result.confusion_matrix.shape == (2, 2)

    def test_metrics_in_valid_range(self):
        """Все метрики должны быть в диапазоне [0, 1]."""
        y_true = pd.Series([0, 0, 0, 1, 1, 1])
        model = MockModel(
            predictions=np.array([0, 1, 0, 1, 0, 1]),
            probabilities=np.array([0.2, 0.6, 0.3, 0.7, 0.4, 0.8]),
        )

        result = evaluate_model(model, pd.DataFrame({"x": range(6)}), y_true)

        assert 0 <= result.precision <= 1
        assert 0 <= result.recall <= 1
        assert 0 <= result.f1 <= 1
        assert 0 <= result.roc_auc <= 1
        assert 0 <= result.pr_auc <= 1


class TestEvaluationResult:
    """Тесты для EvaluationResult."""

    def test_str_representation(self):
        """__str__ должен возвращать читаемое представление."""
        result = EvaluationResult(
            confusion_matrix=np.array([[10, 2], [1, 7]]),
            precision=0.85,
            recall=0.90,
            f1=0.87,
            roc_auc=0.95,
            pr_auc=0.88,
            classification_report="test report",
        )

        str_repr = str(result)

        assert "Precision: 0.8500" in str_repr
        assert "Recall:    0.9000" in str_repr
        assert "F1-Score:  0.8700" in str_repr
        assert "ROC-AUC:   0.9500" in str_repr
        assert "PR-AUC:    0.8800" in str_repr

    def test_handles_none_auc(self):
        """Должен корректно обрабатывать None для AUC метрик."""
        result = EvaluationResult(
            confusion_matrix=np.array([[10, 2], [1, 7]]),
            precision=0.85,
            recall=0.90,
            f1=0.87,
            roc_auc=None,
            pr_auc=None,
            classification_report="test report",
        )

        str_repr = str(result)

        assert "ROC-AUC" not in str_repr
        assert "PR-AUC" not in str_repr
