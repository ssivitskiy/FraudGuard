from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib

from fraudguard.data import load_raw_data, train_valid_test_split
from fraudguard.evaluate import evaluate_model
from fraudguard.features import add_basic_features, build_preprocessor
from fraudguard.models import build_forest_model, build_logreg_model

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fraud detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="transactions.csv",
        help="Filename of the dataset in data/raw/",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="isFraud",
        help="Name of the target column",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["logreg", "forest", "both"],
        default="both",
        help="Model type to train",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fraud_model.joblib",
        help="Output filename for the saved model",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.info("Starting training pipeline")
    logger.info("Arguments: %s", vars(args))

    try:
        df = load_raw_data(args.data)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    df = add_basic_features(df)

    X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(
        df, target_col=args.target
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    logger.info("Features: %d numeric, %d categorical", len(num_cols), len(cat_cols))

    best_model = None
    best_f1 = 0.0

    if args.model in ("logreg", "both"):
        logger.info("=" * 50)
        logger.info("Training Logistic Regression")
        logger.info("=" * 50)

        logreg = build_logreg_model(preprocessor)
        logreg.fit(X_train, y_train)

        logger.info("Evaluating on validation set:")
        result = evaluate_model(logreg, X_valid, y_valid)
        print(result)

        if result.f1 > best_f1:
            best_f1 = result.f1
            best_model = logreg

    if args.model in ("forest", "both"):
        logger.info("=" * 50)
        logger.info("Training Random Forest")
        logger.info("=" * 50)

        preprocessor_rf, _, _ = build_preprocessor(X_train)
        forest = build_forest_model(preprocessor_rf)
        forest.fit(X_train, y_train)

        logger.info("Evaluating on validation set:")
        result = evaluate_model(forest, X_valid, y_valid)
        print(result)

        if result.f1 > best_f1:
            best_f1 = result.f1
            best_model = forest

    if best_model is not None:
        logger.info("=" * 50)
        logger.info("Final evaluation on TEST set")
        logger.info("=" * 50)
        test_result = evaluate_model(best_model, X_test, y_test)
        print(test_result)

        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / args.output
        joblib.dump(best_model, model_path)
        logger.info("Model saved to %s", model_path)

    logger.info("Training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
