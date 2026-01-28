'Main preprocessor orchestrating all preprocessing steps.'

from typing import Optional

import pandas as pd

from src.config import PipelineConfig
from src.preprocessing.base import BaseTransformer
from src.preprocessing.date_transformer import DateTransformer
from src.preprocessing.decimal_handler import DecimalHandler
from src.preprocessing.type_detector import TypeDetector
from src.utils import get_logger


class Preprocessor(BaseTransformer):
    '''
    Main preprocessor that orchestrates all preprocessing steps.

    Steps:
    1. DecimalHandler: Convert Spark Decimal to float64
    2. DateTransformer: Convert dates to days difference
    3. TypeDetector: Detect and convert column types
    '''

    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config
        self.logger = get_logger('preprocessing')

        # Initialize transformers
        self.decimal_handler = DecimalHandler()
        self.date_transformer = DateTransformer(
            report_date_column=config.date_column
        )

        # Columns to exclude from type detection
        exclude_cols = (
            config.get_id_columns_list() +
            [config.date_column, config.target_column]
        )
        if config.get_balance_columns_list():
            exclude_cols.extend(config.get_balance_columns_list())

        self.type_detector = TypeDetector(
            unique_threshold=config.unique_threshold,
            exclude_columns=exclude_cols
        )

        self.feature_columns_: list[str] = []

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> 'Preprocessor':
        'Fit all preprocessing steps.'
        self.logger.info(f'Fitting preprocessor on {len(df):,} rows...')

        # Step 1: Decimal handling
        self.decimal_handler.fit(df)
        df = self.decimal_handler.transform(df)

        # Step 2: Date transformation
        self.date_transformer.fit(df)
        df = self.date_transformer.transform(df)

        # Step 3: Type detection
        self.type_detector.fit(df)

        # Store feature columns
        self._compute_feature_columns(df)

        self.is_fitted_ = True
        self.logger.info(
            f'Preprocessing fitted: {len(self.feature_columns_)} features'
        )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        'Apply all preprocessing steps.'
        self._check_is_fitted()

        self.logger.info(f'Transforming {len(df):,} rows...')

        # Apply transformers in order
        df = self.decimal_handler.transform(df)
        df = self.date_transformer.transform(df)
        df = self.type_detector.transform(df)

        return df

    def _compute_feature_columns(self, df: pd.DataFrame) -> None:
        'Compute list of feature columns (excluding id, date, target, balance).'
        exclude_cols = set(self.config.get_id_columns_list())
        exclude_cols.add(self.config.date_column)
        exclude_cols.add(self.config.target_column)

        if self.config.get_balance_columns_list():
            exclude_cols.update(self.config.get_balance_columns_list())

        self.feature_columns_ = [
            col for col in df.columns if col not in exclude_cols
        ]

    def get_feature_columns(self) -> list[str]:
        'Get list of feature columns.'
        return self.feature_columns_.copy()

    def get_categorical_features(self) -> list[str]:
        'Get categorical feature names for CatBoost.'
        cat_features = self.type_detector.get_catboost_cat_features()
        # Only return features that are in feature_columns
        return [f for f in cat_features if f in self.feature_columns_]

    def get_numeric_features(self) -> list[str]:
        'Get numeric feature names.'
        types = self.type_detector.get_feature_types()
        return [f for f in types['numeric'] if f in self.feature_columns_]

    def get_params(self) -> dict:
        'Get preprocessor parameters for serialization.'
        return {
            'decimal_handler': self.decimal_handler.get_params(),
            'date_transformer': self.date_transformer.get_params(),
            'type_detector': self.type_detector.get_params(),
            'feature_columns': self.feature_columns_
        }

    def set_params(self, params: dict) -> 'Preprocessor':
        'Set preprocessor parameters from serialization.'
        self.decimal_handler.set_params(params.get('decimal_handler', {}))
        self.date_transformer.set_params(params.get('date_transformer', {}))
        self.type_detector.set_params(params.get('type_detector', {}))
        self.feature_columns_ = params.get('feature_columns', [])
        self.is_fitted_ = True
        return self
