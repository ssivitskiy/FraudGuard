"""Общие фикстуры для тестов."""

import pandas as pd
import pytest


@pytest.fixture
def sample_transactions():
    """Создаёт типичный набор транзакций для тестов."""
    return pd.DataFrame({
        "step": [1, 2, 3, 4, 5],
        "type": ["PAYMENT", "CASH_OUT", "TRANSFER", "PAYMENT", "CASH_OUT"],
        "amount": [100.0, 5000.0, 300.0, 50.0, 10000.0],
        "nameOrig": ["C1", "C2", "C3", "C4", "C5"],
        "nameDest": ["M1", "M2", "M3", "M4", "M5"],
        "oldbalanceOrg": [1000, 6000, 500, 100, 15000],
        "newbalanceOrg": [900, 1000, 200, 50, 5000],
        "oldbalanceDest": [0, 0, 0, 0, 0],
        "newbalanceDest": [100, 5000, 300, 50, 10000],
        "isFraud": [0, 1, 0, 0, 1],
    })


@pytest.fixture
def fraud_df():
    """DataFrame с мошенническими транзакциями."""
    return pd.DataFrame({
        "amount": [10000.0, 50000.0],
        "type": ["CASH_OUT", "TRANSFER"],
        "isFraud": [1, 1],
    })


@pytest.fixture
def legit_df():
    """DataFrame с легитимными транзакциями."""
    return pd.DataFrame({
        "amount": [50.0, 100.0, 200.0],
        "type": ["PAYMENT", "PAYMENT", "DEBIT"],
        "isFraud": [0, 0, 0],
    })
