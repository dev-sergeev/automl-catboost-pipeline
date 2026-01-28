'Detect and set column types (numeric vs categorical).'

from typing import Optional

import numpy as np
import pandas as pd

from src.preprocessing.base import BaseTransformer
from src.utils import get_logger


class TypeDetector(BaseTransformer):
    '''
    Detect and convert column types for CatBoost.

    Rules:
    - Numeric columns stay as numeric
    - String columns with unique values > threshold -> treat as numeric (hash)
    - String columns with unique values <= threshold -> category
    - Boolean columns -> category
    '''

    def __init__(
        self,
        unique_threshold: int = 20,
        exclude_columns: Optional[list[str]] = None
    ):
        super().__init__()
        self.unique_threshold = unique_threshold
        self.exclude_columns = exclude_columns or []
        self.logger = get_logger('preprocessing.type_detector')

        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.column_types_: dict[str, str] = {}

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TypeDetector':
        'Detect types for each column.'
        self.numeric_columns_ = []
        self.categorical_columns_ = []
        self.column_types_ = {}

        for col in df.columns:
            if col in self.exclude_columns:
                continue

            col_type = self._detect_column_type(df[col])
            self.column_types_[col] = col_type

            if col_type == 'numeric':
                self.numeric_columns_.append(col)
            elif col_type == 'categorical':
                self.categorical_columns_.append(col)

        self.logger.info(
            f'Detected {len(self.numeric_columns_)} numeric, '
            f'{len(self.categorical_columns_)} categorical columns'
        )

        self._params = {
            'unique_threshold': self.unique_threshold,
            'exclude_columns': self.exclude_columns,
            'column_types': self.column_types_
        }
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        'Convert columns to detected types.'
        self._check_is_fitted()

        df = df.copy()

        for col in df.columns:
            if col in self.exclude_columns or col not in self.column_types_:
                continue

            col_type = self.column_types_[col]

            if col_type == 'numeric':
                df[col] = self._to_numeric(df[col])
            elif col_type == 'categorical':
                df[col] = self._to_categorical(df[col])

        return df

    def _detect_column_type(self, series: pd.Series) -> str:
        'Detect the type of a single column.'
        dtype = series.dtype

        # Boolean (check before numeric since bool is numeric in pandas)
        if pd.api.types.is_bool_dtype(dtype):
            return 'categorical'

        # Already numeric
        if pd.api.types.is_numeric_dtype(dtype):
            return 'numeric'

        # Already category
        if isinstance(dtype, pd.CategoricalDtype):
            return 'categorical'

        # Object/string type - check unique count
        if dtype is object or pd.api.types.is_string_dtype(dtype):
            n_unique = series.nunique()

            if n_unique > self.unique_threshold:
                # High cardinality - try to convert to numeric
                if self._can_convert_to_numeric(series):
                    return 'numeric'
                else:
                    # High cardinality string that can't be numeric
                    # Keep as categorical for CatBoost to handle
                    return 'categorical'
            else:
                return 'categorical'

        # Datetime - should be handled by DateTransformer
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return 'numeric'

        # Default to categorical
        return 'categorical'

    def _can_convert_to_numeric(self, series: pd.Series) -> bool:
        'Check if string column can be converted to numeric.'
        non_null = series.dropna()
        if len(non_null) == 0:
            return True

        try:
            pd.to_numeric(non_null.iloc[:1000], errors='raise')
            return True
        except (ValueError, TypeError):
            return False

    def _to_numeric(self, series: pd.Series) -> pd.Series:
        'Convert series to numeric.'
        if pd.api.types.is_numeric_dtype(series.dtype):
            return series.astype(np.float64)

        return pd.to_numeric(series, errors='coerce').astype(np.float64)

    def _to_categorical(self, series: pd.Series) -> pd.Series:
        'Convert series to categorical (string for CatBoost).'
        # CatBoost handles categories as strings
        return series.fillna('__NAN__').astype(str)

    def get_feature_types(self) -> dict[str, list[str]]:
        'Get lists of numeric and categorical features.'
        return {
            'numeric': self.numeric_columns_.copy(),
            'categorical': self.categorical_columns_.copy()
        }

    def get_catboost_cat_features(self) -> list[str]:
        'Get categorical feature names for CatBoost.'
        return self.categorical_columns_.copy()

    def get_params(self) -> dict:
        'Get transformer parameters.'
        return {
            'unique_threshold': self.unique_threshold,
            'exclude_columns': self.exclude_columns,
            'column_types': self.column_types_,
            'numeric_columns': self.numeric_columns_,
            'categorical_columns': self.categorical_columns_
        }

    def set_params(self, params: dict) -> 'TypeDetector':
        'Set transformer parameters.'
        self.unique_threshold = params.get('unique_threshold', 20)
        self.exclude_columns = params.get('exclude_columns', [])
        self.column_types_ = params.get('column_types', {})
        self.numeric_columns_ = params.get('numeric_columns', [])
        self.categorical_columns_ = params.get('categorical_columns', [])
        self.is_fitted_ = True
        return self
