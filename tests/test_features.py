import pandas as pd

from fraudguard.features import add_basic_features, build_preprocessor


def test_add_basic_features_returns_copy():
    df = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "type": ["PAYMENT", "CASH_OUT", "TRANSFER"],
            "amount": [100.0, 250.5, 300.0],
            "isFraud": [0, 1, 0],
        }
    )

    df2 = add_basic_features(df)

    # исходный датафрейм не должен измениться по ссылке
    assert df is not df2
    # по строкам/столбцам всё совпадает (пока мы ничего не добавляем сверх)
    assert list(df.columns) == list(df2.columns)
    assert df.shape == df2.shape


def test_build_preprocessor_separates_numeric_and_categorical():
    df = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "type": ["PAYMENT", "CASH_OUT", "TRANSFER"],
            "amount": [100.0, 250.5, 300.0],
            "nameOrig": ["C123", "C456", "C789"],
            "isFraud": [0, 1, 0],
        }
    )

    # передаём признаки без таргета
    X = df.drop(columns=["isFraud"])

    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # числовые и категориальные колонки не пересекаются
    assert set(num_cols).isdisjoint(set(cat_cols))

    # базовые проверки, что ожидаемые колонки попали куда надо
    assert "amount" in num_cols
    assert "step" in num_cols
    assert "type" in cat_cols
    assert "nameOrig" in cat_cols

    # таргета нет ни в одной группе
    assert "isFraud" not in num_cols
    assert "isFraud" not in cat_cols

    # сам препроцессор — ColumnTransformer
    from sklearn.compose import ColumnTransformer

    assert isinstance(preprocessor, ColumnTransformer)
