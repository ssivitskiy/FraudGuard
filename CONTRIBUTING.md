# Contributing to FraudGuard

Спасибо за интерес к проекту! Этот документ описывает процесс внесения изменений.

## Быстрый старт

```bash
# 1. Форкните и клонируйте репозиторий
git clone https://github.com/YOUR_USERNAME/fraudguard.git
cd fraudguard

# 2. Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Установите зависимости для разработки
make install-dev
# или: pip install -e ".[all]"

# 4. Установите pre-commit хуки
make pre-commit
# или: pre-commit install

# 5. Создайте ветку для изменений
git checkout -b feature/your-feature-name
```

## Структура проекта

```
FraudGuard/
├── fraudguard/          # Основной пакет
│   ├── data.py          # Загрузка и разбиение данных
│   ├── features.py      # Инженерия признаков
│   ├── models.py        # Определения моделей
│   └── evaluate.py      # Метрики и оценка
├── scripts/             # CLI-скрипты
│   ├── train.py         # Обучение модели
│   └── predict.py       # Инференс
├── app/                 # Streamlit приложение
├── tests/               # Тесты
├── notebooks/           # Jupyter ноутбуки для EDA
└── data/                # Данные (не в git)
```

## Процесс разработки

### 1. Перед началом работы

- Убедитесь, что issue создан для вашей задачи
- Обсудите крупные изменения в issue перед реализацией

### 2. Написание кода

**Стиль кода:**
- Следуйте PEP 8 (проверяется автоматически через ruff)
- Используйте type hints для всех публичных функций
- Документируйте функции с помощью docstrings (Google style)

```python
def calculate_fraud_probability(
    transaction: pd.DataFrame,
    model: Pipeline,
) -> float:
    """Вычисляет вероятность мошенничества для транзакции.

    Args:
        transaction: DataFrame с одной строкой — данные транзакции.
        model: Обученная модель-пайплайн.

    Returns:
        Вероятность мошенничества от 0 до 1.

    Raises:
        ValueError: Если transaction содержит более одной строки.
    """
    ...
```

**Проверка кода:**
```bash
make check  # Запускает lint + type-check + test
```

### 3. Тестирование

- Покройте новый код тестами (минимум 70% coverage)
- Запускайте тесты перед коммитом:

```bash
make test        # Полный прогон с coverage
make test-fast   # Быстрый прогон без coverage
```

### 4. Коммиты

Используйте [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: добавить поддержку XGBoost модели
fix: исправить обработку пустых значений в amount
docs: обновить README с примерами использования
test: добавить тесты для feature engineering
refactor: вынести препроцессинг в отдельный модуль
```

### 5. Pull Request

1. Убедитесь, что все проверки проходят (`make check`)
2. Обновите документацию при необходимости
3. Заполните шаблон PR
4. Запросите review

## Запуск тестов

```bash
# Все тесты с coverage
pytest --cov=fraudguard --cov-report=term-missing

# Конкретный файл
pytest tests/test_features.py -v

# Конкретный тест
pytest tests/test_features.py::test_add_basic_features -v
```

## Добавление новой модели

1. Добавьте функцию-конструктор в `fraudguard/models.py`:

```python
def build_xgboost_model(preprocessor) -> Pipeline:
    """Создаёт XGBoost классификатор."""
    clf = XGBClassifier(...)
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])
```

2. Добавьте тест в `tests/test_models.py`
3. Обновите `scripts/train.py` для поддержки новой модели

## Вопросы?

Создайте issue с тегом `question` или напишите в Discussions.