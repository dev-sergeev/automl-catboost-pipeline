'Filter features based on PSI stability over time periods.'

from typing import Optional

import numpy as np
import pandas as pd

from src.feature_selection.base import BaseSelector
from src.feature_selection.psi_filter import PSIFilter
from src.utils import get_logger


class PSITimeFilter(BaseSelector):
    '''
    Remove features with high PSI drift over time.

    Splits data into time periods and compares each period
    against the first (earliest) period. If PSI exceeds threshold
    in any period, the feature is removed.

    This helps identify features that are unstable over time,
    even if they appear stable in a single train/valid comparison.

    Speed: 2 (medium-fast)
    '''

    def __init__(
        self,
        date_column: str,
        threshold: float = 0.25,
        period_days: int = 14,
        n_bins: int = 10,
        min_period_days: int = 14
    ):
        '''
        Initialize PSITimeFilter.

        Args:
            date_column: Name of the date column for time splitting
            threshold: PSI threshold for removal (default 0.25)
            period_days: Days per period for splitting (default 14)
            n_bins: Number of bins for PSI calculation (default 10)
            min_period_days: Minimum days in last period, otherwise merge
                            with previous (default 14)
        '''
        super().__init__()
        self.date_column = date_column
        self.threshold = threshold
        self.period_days = period_days
        self.n_bins = n_bins
        self.min_period_days = min_period_days
        self.logger = get_logger('feature_selection.psi_time')

        self.psi_values_: dict[str, dict[str, float]] = {}  # {feature: {period: psi}}
        self.max_psi_values_: dict[str, float] = {}  # {feature: max_psi}
        self.period_ranges_: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None
    ) -> 'PSITimeFilter':
        '''
        Compute PSI across time periods and filter features.

        Args:
            X: Training features (must include date_column)
            y: Training target (not used)
            sample_weight: Sample weights (not used)
            X_valid: Validation features to combine with X
            y_valid: Validation target (not used)
        '''
        # Combine train and valid if provided
        if X_valid is not None:
            X_combined = pd.concat([X, X_valid], ignore_index=True)
        else:
            X_combined = X.copy()

        if self.date_column not in X_combined.columns:
            raise ValueError(f'Date column "{self.date_column}" not found in data')

        self.logger.info(
            f'Fitting PSITimeFilter (threshold={self.threshold}, '
            f'period={self.period_days} days) on {X_combined.shape[1]} features...'
        )

        # Split data into time periods
        periods = self._split_by_periods(X_combined)

        if len(periods) < 2:
            self.logger.warning(
                'Not enough time periods for PSI comparison. '
                'Keeping all features.'
            )
            feature_cols = [c for c in X.columns if c != self.date_column]
            self.selected_features_ = feature_cols
            self.removed_features_ = []
            self.is_fitted_ = True
            return self

        self.logger.info(f'Split data into {len(periods)} time periods')

        # First period is reference
        reference_data = periods[0]
        comparison_periods = periods[1:]

        # Feature columns (exclude date)
        feature_cols = [c for c in X_combined.columns if c != self.date_column]

        # PSI calculator
        psi_calc = PSIFilter(threshold=self.threshold, n_bins=self.n_bins)

        self.selected_features_ = []
        self.removed_features_ = []
        self.psi_values_ = {}
        self.max_psi_values_ = {}

        for col in feature_cols:
            col_psi = {}
            max_psi = 0.0

            for i, period_data in enumerate(comparison_periods):
                period_name = f'period_{i + 2}'

                # Compute PSI between reference and this period
                psi = psi_calc._compute_psi(
                    reference_data[col],
                    period_data[col]
                )
                col_psi[period_name] = psi
                max_psi = max(max_psi, psi)

            self.psi_values_[col] = col_psi
            self.max_psi_values_[col] = max_psi

            if max_psi <= self.threshold:
                self.selected_features_.append(col)
            else:
                self.removed_features_.append(col)

        self.logger.info(
            f'Removed {len(self.removed_features_)} features with '
            f'max PSI > {self.threshold} across time periods'
        )

        self._params = {
            'date_column': self.date_column,
            'threshold': self.threshold,
            'period_days': self.period_days,
            'n_bins': self.n_bins,
            'psi_values': self.psi_values_,
            'max_psi_values': self.max_psi_values_,
            'period_ranges': [
                (str(start), str(end)) for start, end in self.period_ranges_
            ]
        }
        self.is_fitted_ = True
        return self

    def _split_by_periods(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        '''Split DataFrame into time periods.'''
        # Ensure date column is datetime
        dates = pd.to_datetime(df[self.date_column])

        min_date = dates.min()
        max_date = dates.max()
        total_days = (max_date - min_date).days

        if total_days < self.period_days:
            self.logger.warning(
                f'Total date range ({total_days} days) is less than '
                f'period_days ({self.period_days}). Cannot split.'
            )
            return [df]

        # Create period boundaries
        period_starts = []
        current_start = min_date

        while current_start < max_date:
            period_starts.append(current_start)
            current_start = current_start + pd.Timedelta(days=self.period_days)

        # Check if last period is too short
        if len(period_starts) > 1:
            last_period_start = period_starts[-1]
            last_period_days = (max_date - last_period_start).days

            if last_period_days < self.min_period_days:
                # Merge last period with previous
                period_starts = period_starts[:-1]
                self.logger.info(
                    f'Merged last period ({last_period_days} days) '
                    f'with previous period'
                )

        # Split data by periods
        periods = []
        self.period_ranges_ = []

        for i, start in enumerate(period_starts):
            if i < len(period_starts) - 1:
                end = period_starts[i + 1]
                mask = (dates >= start) & (dates < end)
            else:
                end = max_date + pd.Timedelta(days=1)
                mask = dates >= start

            period_data = df[mask].copy()
            if len(period_data) > 0:
                periods.append(period_data)
                self.period_ranges_.append((start, end - pd.Timedelta(days=1)))

        return periods

    def get_psi_values(self) -> dict[str, dict[str, float]]:
        '''Get PSI values for all features across all periods.'''
        return self.psi_values_.copy()

    def get_max_psi_values(self) -> dict[str, float]:
        '''Get maximum PSI value for each feature.'''
        return self.max_psi_values_.copy()

    def get_period_ranges(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        '''Get date ranges for each period.'''
        return self.period_ranges_.copy()

    def get_psi_dataframe(self) -> pd.DataFrame:
        '''Get PSI values as DataFrame for analysis.'''
        if not self.psi_values_:
            return pd.DataFrame()

        rows = []
        for feature, periods in self.psi_values_.items():
            row = {'feature': feature, 'max_psi': self.max_psi_values_[feature]}
            row.update(periods)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values('max_psi', ascending=False)
        return df

    def get_params(self) -> dict:
        '''Get filter parameters.'''
        return {
            'date_column': self.date_column,
            'threshold': self.threshold,
            'period_days': self.period_days,
            'n_bins': self.n_bins,
            'min_period_days': self.min_period_days,
            'psi_values': self.psi_values_,
            'max_psi_values': self.max_psi_values_,
            'period_ranges': [
                (str(start), str(end)) for start, end in self.period_ranges_
            ],
            'selected_features': self.selected_features_,
            'removed_features': self.removed_features_
        }

    def set_params(self, params: dict) -> 'PSITimeFilter':
        '''Set filter parameters.'''
        self.date_column = params.get('date_column', 'report_date')
        self.threshold = params.get('threshold', 0.25)
        self.period_days = params.get('period_days', 14)
        self.n_bins = params.get('n_bins', 10)
        self.min_period_days = params.get('min_period_days', 14)
        self.psi_values_ = params.get('psi_values', {})
        self.max_psi_values_ = params.get('max_psi_values', {})

        # Parse period ranges if stored as strings
        period_ranges = params.get('period_ranges', [])
        self.period_ranges_ = [
            (pd.Timestamp(start), pd.Timestamp(end))
            for start, end in period_ranges
        ]

        self.selected_features_ = params.get('selected_features', [])
        self.removed_features_ = params.get('removed_features', [])
        self.is_fitted_ = True
        return self
