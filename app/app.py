import joblib
from pathlib import Path
import pandas as pd
import streamlit as st

from fraudguard.features import add_basic_features


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "fraud_model.joblib"
    return joblib.load(model_path)


def main():
    st.title("FraudGuard: детектор мошеннических транзакций")

    amount = st.number_input("Сумма транзакции", min_value=0.0, value=100.0, step=1.0)
    transaction_type = st.selectbox("Тип транзакции", ["online", "offline", "atm"])
    device_type = st.selectbox("Тип устройства", ["mobile", "web", "pos-terminal"])
    transaction_time = st.text_input("Время транзакции", "2025-01-01 12:00:00")

    if st.button("Проверить"):
        model = load_model()

        row = {
            "amount": amount,
            "transaction_type": transaction_type,
            "device_type": device_type,
            "transaction_time": transaction_time,
        }

        df = pd.DataFrame([row])
        df = add_basic_features(df)

        proba = model.predict_proba(df)[:, 1][0]
        pred = int(proba >= 0.5)

        st.write(f"**Вероятность мошенничества:** {proba:.4f}")
        if pred == 1:
            st.error("⚠️ Высокий риск мошенничества")
        else:
            st.success("✅ Транзакция выглядит безопасной")


if __name__ == "__main__":
    main()
