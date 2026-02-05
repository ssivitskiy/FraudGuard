from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

from fraudguard.features import add_basic_features

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict fraud probability for a single transaction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--amount",
        type=float,
        required=True,
        help="Transaction amount",
    )
    parser.add_argument(
        "--transaction_type",
        type=str,
        required=True,
        help="Transaction type (e.g., PAYMENT, CASH_OUT, TRANSFER)",
    )
    parser.add_argument(
        "--device_type",
        type=str,
        required=True,
        help="Device type (e.g., mobile, web, pos-terminal)",
    )
    parser.add_argument(
        "--transaction_time",
        type=str,
        required=True,
        help="Transaction timestamp (e.g., '2025-01-01 12:34:56')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fraud_model.joblib",
        help="Model filename in models/ directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for fraud classification",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = MODELS_DIR / args.model
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        logger.error("Please run 'python -m scripts.train' first")
        return 1

    model = joblib.load(model_path)
    logger.info("Loaded model from %s", model_path)

    row = {
        "amount": args.amount,
        "transaction_type": args.transaction_type,
        "device_type": args.device_type,
        "transaction_time": args.transaction_time,
    }

    df = pd.DataFrame([row])
    df = add_basic_features(df)

    proba = model.predict_proba(df)[:, 1][0]
    pred = int(proba >= args.threshold)

    result = {
        "fraud_probability": round(proba, 4),
        "prediction": pred,
        "is_fraud": pred == 1,
        "threshold": args.threshold,
        "input": row,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("\n" + "=" * 40)
        print("FraudGuard Prediction Result")
        print("=" * 40)
        print(f"Amount:            ${args.amount:,.2f}")
        print(f"Transaction type:  {args.transaction_type}")
        print(f"Device type:       {args.device_type}")
        print(f"Time:              {args.transaction_time}")
        print("-" * 40)
        print(f"Fraud probability: {proba:.4f} ({proba*100:.1f}%)")
        print(f"Prediction:        {'🚨 FRAUD' if pred == 1 else '✅ LEGITIMATE'}")
        print("=" * 40 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
