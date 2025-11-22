import argparse
import joblib
from pathlib import Path
import pandas as pd

from fraudguard.features import add_basic_features


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amount", type=float, required=True)
    parser.add_argument("--transaction_type", type=str, required=True)
    parser.add_argument("--device_type", type=str, required=True)
    parser.add_argument("--transaction_time", type=str, required=True)
    args = parser.parse_args()

    model_path = MODELS_DIR / "fraud_model.joblib"
    model = joblib.load(model_path)

    row = {
        "amount": args.amount,
        "transaction_type": args.transaction_type,
        "device_type": args.device_type,
        "transaction_time": args.transaction_time,
    }

    df = pd.DataFrame([row])
    df = add_basic_features(df)

    proba = model.predict_proba(df)[:, 1][0]
    pred = int(proba >= 0.5)

    print(f"Fraud probability: {proba:.4f}")
    print(f"Prediction (0=not fraud, 1=fraud): {pred}")


if __name__ == "__main__":
    main()
