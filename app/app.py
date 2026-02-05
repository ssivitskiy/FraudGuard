from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from fraudguard.features import add_basic_features

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

st.set_page_config(
    page_title="FraudGuard",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "fraud_model.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def main():
    st.title("🛡️ FraudGuard")
    st.markdown("**Детектор мошеннических транзакций**")
    st.markdown("---")

    model = load_model()
    if model is None:
        st.error("⚠️ Модель не найдена! Запустите `python -m scripts.train` для обучения модели.")
        st.stop()

    st.subheader("📝 Параметры транзакции")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "💰 Сумма транзакции",
            min_value=0.0,
            max_value=10_000_000.0,
            value=100.0,
            step=10.0,
            help="Введите сумму операции",
        )

        transaction_type = st.selectbox(
            "📋 Тип транзакции",
            options=["PAYMENT", "CASH_OUT", "TRANSFER", "DEBIT", "CASH_IN"],
            help="Выберите тип платёжной операции",
        )

    with col2:
        device_type = st.selectbox(
            "📱 Тип устройства",
            options=["mobile", "web", "pos-terminal", "atm"],
            help="Устройство, с которого совершена операция",
        )

        transaction_date = st.date_input(
            "📅 Дата",
            value=datetime.now().date(),
        )
        transaction_time_input = st.time_input(
            "🕐 Время",
            value=datetime.now().time(),
        )

    transaction_time = f"{transaction_date} {transaction_time_input}"

    st.markdown("---")

    if st.button("🔍 Проверить транзакцию", type="primary", use_container_width=True):
        with st.spinner("Анализ транзакции..."):
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

        st.markdown("---")
        st.subheader("📊 Результат анализа")

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.metric(
                label="Вероятность мошенничества",
                value=f"{proba:.1%}",
                delta=None,
            )

            st.progress(proba)

        with col_right:
            if pred == 1:
                st.error("🚨 **ВЫСОКИЙ РИСК**")
            elif proba > 0.3:
                st.warning("⚠️ **СРЕДНИЙ РИСК**")
            else:
                st.success("✅ **НИЗКИЙ РИСК**")

        with st.expander("📋 Детали транзакции"):
            st.json(row)

    with st.sidebar:
        st.markdown("### ℹ️ О приложении")
        st.markdown(
            """
            **FraudGuard** использует машинное обучение
            для анализа транзакций и выявления подозрительной активности.

            **Модель учитывает:**
            - Сумму операции
            - Тип транзакции
            - Устройство
            - Время проведения

            **Пороги риска:**
            - 🟢 < 30% — низкий риск
            - 🟡 30-50% — средний риск
            - 🔴 > 50% — высокий риск
            """
        )


if __name__ == "__main__":
    main()
