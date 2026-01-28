'Transform date columns to days difference from report_date.'

from typing import Optional

import numpy as np
import pandas as pd

from src.preprocessing.base import BaseTransformer
from src.utils import get_logger


class DateTransformer(BaseTransformer):
    '''
    Transform date columns to numeric (days difference from report_date).

    Date columns are converted to the number of days between the date
    and the report_date column. This creates meaningful numeric features
    that represent "time since" or "time until" events.
    '''

    def __init__(self, report_date_column: str = 'report_date'):
        super().__init__()
        self.report_date_column = report_date_column
        self.logger = get_logger('preprocessing.date')
        self.date_columns_: list[str] = []

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DateTransformer':
        'Identify date columns (excluding report_date).'
        self.date_columns_ = []

        for col in df.columns:
            if col == self.report_date_column:
                continue
            if self._is_date_column(df[col]):
                self.date_columns_.append(col)

        if self.date_columns_:
            self.logger.info(
                f'Found {len(self.date_columns_)} date columns to transform'
            )

        self._params = {
            'report_date_column': self.report_date_column,
            'date_columns': self.date_columns_
        }
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        'Transform date columns to days difference.'
        self._check_is_fitted()

        if not self.date_columns_:
            return df

        if self.report_date_column not in df.columns:
            self.logger.warning(
                f'Report date column "{self.report_date_column}" not found, '
                'skipping date transformation'
            )
            return df

        df = df.copy()

        # Ensure report_date is datetime
        report_date = pd.to_datetime(df[self.report_date_column])

        for col in self.date_columns_:
            if col in df.columns:
                df[col] = self._transform_date_column(df[col], report_date, col)

        self.logger.info(
            f'Transformed {len(self.date_columns_)} date columns to days difference'
        )

        return df

    def _is_date_column(self, series: pd.Series) -> bool:
        'Check if series is a date/datetime column.'
        # Check if already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        # Check if object or string type that can be parsed as datetime
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                sample = non_null.iloc[:100]
                try:
                    pd.to_datetime(sample, errors='raise')
                    return True
                except (ValueError, TypeError):
                    return False

        return False

    def _transform_date_column(
        self,
        series: pd.Series,
        report_date: pd.Series,
        col_name: str
    ) -> pd.Series:
        'Convert date column to days difference from report_date.'
        try:
            date_values = pd.to_datetime(series, errors='coerce')
            days_diff = (date_values - report_date).dt.days
            return days_diff.astype(np.float64)
        except Exception as e:
            self.logger.warning(
                f'Could not transform date column "{col_name}": {e}'
            )
            return series

    def get_params(self) -> dict:
        'Get transformer parameters.'
        return {
            'report_date_column': self.report_date_column,
            'date_columns': self.date_columns_
        }

    def set_params(self, params: dict) -> 'DateTransformer':
        'Set transformer parameters.'
        self.report_date_column = params.get('report_date_column', 'report_date')
        self.date_columns_ = params.get('date_columns', [])
        self.is_fitted_ = True
        return self
