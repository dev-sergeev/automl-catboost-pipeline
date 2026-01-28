# CatBoost Pipeline

Универсальный пайплайн для обучения моделей градиентного бустинга (CatBoost) для скоринговых задач.

## Возможности

- **Бинарная классификация** для датасетов 1-10 млн строк, 500+ признаков
- **Group-aware разделение данных**: один клиент не может быть в разных сплитах (train/valid/OOS/OOT)
- **Балансировка по группам**: опциональное взвешивание по столбцу (например, по типу продукта)
- **Автоматический препроцессинг**: обработка Decimal, преобразование дат, определение типов
- **Каскадный отбор признаков**: 7 методов от быстрых к медленным
- **Оптимизация гиперпараметров**: Optuna с ранней остановкой
- **Сериализация артефактов**: сохранение и загрузка модели для скоринга
- **Визуализация**: графики метрик по сплитам и динамика по времени

## Установка

```bash
# Клонирование репозитория
git clone <repository-url>
cd AutoML

# Установка зависимостей
pip install -e .

# Для разработки (с тестами)
pip install -e ".[dev]"
```

## Быстрый старт

### Обучение модели

```python
from src.config import PipelineConfig, FeatureSelectionConfig
from src.pipeline import CatBoostPipeline

# Конфигурация
config = PipelineConfig(
    id_columns=['client_id', 'application_id'],
    date_column='report_date',
    target_column='target',
    client_column='client_id',  # Для group-aware split
    random_seed=42,
    n_trials=50,
    artifacts_dir='./artifacts'
)

# Обучение
pipeline = CatBoostPipeline(config)
pipeline.fit(df, run_optimization=True, save_artifacts=True)

# Результаты
metrics = pipeline.get_metrics()
print(f"Valid Gini: {metrics['valid']['gini']:.4f}")
```

### Скоринг новых данных

```python
from src.scoring import Scorer

# Загрузка модели
scorer = Scorer('./artifacts')
scorer.load()

# Скоринг
scores = scorer.score(new_data)
```

## Структура проекта

```
AutoML/
├── pyproject.toml              # Зависимости
├── README.md
├── notebooks/
│   ├── 01_training_pipeline.ipynb   # Обучение модели
│   └── 02_scoring.ipynb             # Скоринг новых данных
├── src/
│   ├── config.py               # Конфигурация (dataclasses)
│   ├── utils.py                # Утилиты
│   ├── pipeline.py             # Главный оркестратор
│   ├── data/
│   │   ├── loader.py           # Загрузка Spark parquet
│   │   ├── splitter.py         # Train/Valid/OOS/OOT split
│   │   └── balancer.py         # Вычисление sample_weight
│   ├── preprocessing/
│   │   ├── decimal_handler.py  # Spark Decimal -> float64
│   │   ├── date_transformer.py # Даты -> разница в днях
│   │   ├── type_detector.py    # Определение типов (num/cat)
│   │   └── preprocessor.py     # Оркестратор
│   ├── feature_selection/
│   │   ├── missing_filter.py   # Фильтр пропусков
│   │   ├── variance_filter.py  # Фильтр дисперсии
│   │   ├── correlation_filter.py # Фильтр корреляций
│   │   ├── psi_filter.py       # PSI фильтр
│   │   ├── importance_filter.py  # Feature Importance
│   │   ├── backward_selection.py # Backward Selection
│   │   ├── forward_selection.py  # Forward Selection
│   │   └── selector.py         # Оркестратор отбора
│   ├── optimization/
│   │   ├── search_space.py     # Пространство гиперпараметров
│   │   └── optimizer.py        # Optuna оптимизатор
│   ├── scoring/
│   │   ├── artifacts.py        # Сохранение/загрузка артефактов
│   │   └── scorer.py           # Скоринг новых данных
│   ├── evaluation/
│   │   └── metrics.py          # Gini, AUC, метрики по группам
│   └── visualization/
│       ├── metrics_plots.py    # Графики метрик
│       └── time_dynamics.py    # Динамика по месяцам
└── tests/                      # Тесты (pytest)
```

## Конфигурация

### PipelineConfig

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `id_columns` | `str \| list[str]` | - | ID наблюдения |
| `date_column` | `str` | `'report_date'` | Колонка с датой |
| `target_column` | `str` | `'target'` | Целевая переменная |
| `client_column` | `str \| None` | `None` | Колонка клиента для group-aware split |
| `balance_columns` | `str \| list[str] \| None` | `None` | Колонки для балансировки |
| `train_ratio` | `float` | `0.6` | Доля train |
| `valid_ratio` | `float` | `0.2` | Доля valid |
| `oos_ratio` | `float` | `0.1` | Доля OOS |
| `oot_ratio` | `float` | `0.1` | Доля OOT |
| `missing_threshold` | `float` | `0.95` | Порог пропусков |
| `correlation_threshold` | `float` | `0.95` | Порог корреляции |
| `psi_threshold` | `float` | `0.25` | Порог PSI |
| `n_trials` | `int` | `100` | Число trials Optuna |
| `optuna_timeout` | `int` | `3600` | Таймаут Optuna (сек) |
| `artifacts_dir` | `str` | `'./artifacts'` | Путь для артефактов |

### FeatureSelectionConfig

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `run_missing_filter` | `bool` | `True` | Фильтр пропусков |
| `run_variance_filter` | `bool` | `True` | Фильтр дисперсии |
| `run_correlation_filter` | `bool` | `True` | Фильтр корреляций |
| `run_psi_filter` | `bool` | `True` | PSI фильтр |
| `run_importance_filter` | `bool` | `True` | Feature Importance |
| `run_backward_selection` | `bool` | `False` | Backward Selection |
| `run_forward_selection` | `bool` | `False` | Forward Selection |

## Отбор признаков

Методы применяются последовательно от быстрых к медленным:

| # | Метод | Скорость | Описание |
|---|-------|----------|----------|
| 1 | Missing Filter | Быстрый | Удаляет признаки с >95% пропусков |
| 2 | Variance Filter | Быстрый | Удаляет признаки с нулевой дисперсией |
| 3 | Correlation Filter | Средний | Удаляет один из пары при \|corr\| > 0.95 |
| 4 | PSI Filter | Средний | Удаляет нестабильные признаки (PSI > 0.25) |
| 5 | Importance Filter | Средний | Удаляет признаки с нулевой важностью |
| 6 | Backward Selection | Медленный | Итеративно удаляет 10% худших |
| 7 | Forward Selection | Очень медленный | Итеративно добавляет 10% лучших |

## Разделение данных

Данные разделяются на 4 части:

- **Train (60%)**: Обучение модели
- **Valid (20%)**: Early stopping, отбор признаков
- **OOS (10%)**: Out-of-Sample - финальная оценка
- **OOT (10%)**: Out-of-Time - последние даты

**Group-aware split**: один клиент не может быть в разных сплитах. Разделение происходит по уникальным клиентам, а не по наблюдениям.

## Балансировка по группам

При сильном дисбалансе по продуктам (ПК 80%, КК 18%, ИК 2%) модель может терять качество на редких группах. Решение - балансировка через `sample_weight`:

```python
config = PipelineConfig(
    # ...
    balance_columns=['product_type'],  # Балансировка по типу продукта
)
```

Веса вычисляются обратно пропорционально частоте группы.

## Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# С покрытием
pytest tests/ -v --cov=src

# Конкретный модуль
pytest tests/src/feature_selection/ -v
```

## Notebooks

1. **01_training_pipeline.ipynb** - полный цикл обучения модели:
   - Загрузка данных
   - Конфигурация пайплайна
   - Обучение с оптимизацией
   - Визуализация результатов

2. **02_scoring.ipynb** - скоринг новых данных:
   - Загрузка сохраненной модели
   - Скоринг новых наблюдений
   - Анализ распределения скоров

## Зависимости

- Python >= 3.10
- catboost >= 1.2
- optuna >= 3.0
- pandas >= 2.0
- numpy >= 1.24
- scikit-learn >= 1.3
- pyarrow >= 14.0
- matplotlib >= 3.7
- seaborn >= 0.12

## Лицензия

MIT
