'Handler for Spark Decimal types.'

from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd

from src.preprocessing.base import BaseTransformer
from src.utils import get_logger


class DecimalHandler(BaseTransformer):
    '''
    Convert Spark Decimal columns to float64.

    Spark parquet files often contain Decimal types that need to be
    converted to float64 for processing with pandas/numpy.
    '''

    def __init__(self):
        super().__init__()
        self.logger = get_logger('preprocessing.decimal')
        self.decimal_columns_: list[str] = []

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DecimalHandler':
        'Identify columns with Decimal type.'
        self.decimal_columns_ = []

        for col in df.columns:
            if self._is_decimal_column(df[col]):
                self.decimal_columns_.append(col)

        if self.decimal_columns_:
            self.logger.info(
                f'Found {len(self.decimal_columns_)} Decimal columns'
            )

        self._params = {'decimal_columns': self.decimal_columns_}
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        'Convert Decimal columns to float64.'
        self._check_is_fitted()

        if not self.decimal_columns_:
            return df

        df = df.copy()

        for col in self.decimal_columns_:
            if col in df.columns:
                df[col] = self._convert_decimal_column(df[col])

        self.logger.info(
            f'Converted {len(self.decimal_columns_)} Decimal columns to float64'
        )

        return df

    def _is_decimal_column(self, series: pd.Series) -> bool:
        'Check if series contains Decimal objects.'
        # Check dtype
        if series.dtype == object:
            # Sample non-null values to check type
            non_null = series.dropna()
            if len(non_null) > 0:
                sample = non_null.iloc[:100]
                return any(isinstance(v, Decimal) for v in sample)
        return False

    def _convert_decimal_column(self, series: pd.Series) -> pd.Series:
        'Convert Decimal series to float64.'
        return series.apply(
            lambda x: float(x) if isinstance(x, Decimal) else x
        ).astype(np.float64)

    def get_params(self) -> dict:
        'Get transformer parameters.'
        return {'decimal_columns': self.decimal_columns_}

    def set_params(self, params: dict) -> 'DecimalHandler':
        'Set transformer parameters.'
        self.decimal_columns_ = params.get('decimal_columns', [])
        self.is_fitted_ = True
        return self
