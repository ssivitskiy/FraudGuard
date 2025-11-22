import joblib
from pathlib import Path

from fraudguard.data import load_raw_data, train_valid_test_split
from fraudguard.features import add_basic_features, build_preprocessor
from fraudguard.models import build_logreg_model, build_forest_model
from fraudguard.evaluate import evaluate_model


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main():
    df = load_raw_data("transactions.csv")
    df = add_basic_features(df)

    X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(
        df, target_col="isFraud"
    )

    preprocessor, _, _ = build_preprocessor(X_train)

    logreg = build_logreg_model(preprocessor)
    logreg.fit(X_train, y_train)

    print("Logistic Regression on valid:")
    evaluate_model(logreg, X_valid, y_valid)

    forest = build_forest_model(preprocessor)
    forest.fit(X_train, y_train)

    print("Random Forest on valid:")
    evaluate_model(forest, X_valid, y_valid)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(logreg, MODELS_DIR / "fraud_model.joblib")
    print(f"Model saved to {MODELS_DIR / 'fraud_model.joblib'}")


if __name__ == "__main__":
    main()
