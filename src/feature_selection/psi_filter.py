'Filter features with high Population Stability Index (PSI).'

from typing import Optional

import numpy as np
import pandas as pd

from src.feature_selection.base import BaseSelector
from src.utils import get_logger


class PSIFilter(BaseSelector):
    '''
    Remove features with high PSI (Population Stability Index).

    PSI measures distribution shift between train and validation sets.
    High PSI indicates unstable features that may not generalize well.

    Interpretation:
    - PSI < 0.10: No significant shift
    - 0.10 <= PSI < 0.25: Moderate shift
    - PSI >= 0.25: Significant shift (remove)

    Speed: 2 (medium-fast)
    '''

    def __init__(self, threshold: float = 0.25, n_bins: int = 10):
        super().__init__()
        self.threshold = threshold
        self.n_bins = n_bins
        self.logger = get_logger('feature_selection.psi')
        self.psi_values_: dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        X_reference: Optional[pd.DataFrame] = None
    ) -> 'PSIFilter':
        '''
        Compute PSI between X and reference distribution.

        Args:
            X: Current data (e.g., validation set)
            y: Target (not used for PSI)
            sample_weight: Sample weights (not used)
            X_reference: Reference data (e.g., train set). If None, splits X.
        '''
        self.logger.info(
            f'Fitting PSIFilter (threshold={self.threshold}) '
            f'on {X.shape[1]} features...'
        )

        self.selected_features_ = []
        self.removed_features_ = []
        self.psi_values_ = {}

        # If no reference, split X into two halves
        if X_reference is None:
            mid = len(X) // 2
            X_reference = X.iloc[:mid]
            X = X.iloc[mid:]

        for col in X.columns:
            psi = self._compute_psi(X_reference[col], X[col])
            self.psi_values_[col] = psi

            if psi <= self.threshold:
                self.selected_features_.append(col)
            else:
                self.removed_features_.append(col)

        self.logger.info(
            f'Removed {len(self.removed_features_)} features with '
            f'PSI > {self.threshold}'
        )

        self._params = {
            'threshold': self.threshold,
            'n_bins': self.n_bins,
            'psi_values': self.psi_values_
        }
        self.is_fitted_ = True
        return self

    def _compute_psi(
        self,
        expected: pd.Series,
        actual: pd.Series,
        eps: float = 1e-6
    ) -> float:
        'Compute PSI between expected and actual distributions.'
        # Handle categorical features
        if not pd.api.types.is_numeric_dtype(expected.dtype):
            return self._compute_psi_categorical(expected, actual, eps)

        # Numeric features: use quantile-based binning
        return self._compute_psi_numeric(expected, actual, eps)

    def _compute_psi_numeric(
        self,
        expected: pd.Series,
        actual: pd.Series,
        eps: float
    ) -> float:
        'Compute PSI for numeric features using symmetric binning.'
        # Separate NaN values (will be counted as separate bin)
        expected_nan_count = expected.isna().sum()
        actual_nan_count = actual.isna().sum()

        expected_clean = expected.dropna()
        actual_clean = actual.dropna()

        # If both arrays are completely empty, no shift
        if len(expected_clean) == 0 and len(actual_clean) == 0:
            return 0.0

        # Use common range for symmetric PSI
        if len(expected_clean) > 0 and len(actual_clean) > 0:
            data_min = min(expected_clean.min(), actual_clean.min())
            data_max = max(expected_clean.max(), actual_clean.max())
        elif len(expected_clean) > 0:
            data_min, data_max = expected_clean.min(), expected_clean.max()
        else:
            data_min, data_max = actual_clean.min(), actual_clean.max()

        data_range = (data_min, data_max)

        # Compute histograms with common range
        expected_counts = np.histogram(expected_clean, bins=self.n_bins, range=data_range)[0]
        actual_counts = np.histogram(actual_clean, bins=self.n_bins, range=data_range)[0]

        # Add NaN as separate bin
        expected_counts = np.append(expected_counts, expected_nan_count)
        actual_counts = np.append(actual_counts, actual_nan_count)

        # Compute fractions
        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)

        # Replace zeros with small value to avoid log(0)
        expected_pct = np.where(expected_pct == 0, eps, expected_pct)
        actual_pct = np.where(actual_pct == 0, eps, actual_pct)

        # PSI formula
        psi = np.sum((expected_pct - actual_pct) * np.log(expected_pct / actual_pct))

        return float(psi)

    def _compute_psi_categorical(
        self,
        expected: pd.Series,
        actual: pd.Series,
        eps: float
    ) -> float:
        'Compute PSI for categorical features (NaN treated as separate category).'
        # Fill NaN with placeholder to count as separate category
        expected_filled = expected.fillna('__NAN__')
        actual_filled = actual.fillna('__NAN__')

        # Get value counts as proportions
        expected_counts = expected_filled.value_counts(normalize=True)
        actual_counts = actual_filled.value_counts(normalize=True)

        # Get all categories
        all_categories = set(expected_counts.index) | set(actual_counts.index)

        psi = 0.0
        for cat in all_categories:
            expected_pct = expected_counts.get(cat, 0)
            actual_pct = actual_counts.get(cat, 0)

            # Replace zeros with small value
            expected_pct = expected_pct if expected_pct > 0 else eps
            actual_pct = actual_pct if actual_pct > 0 else eps

            psi += (expected_pct - actual_pct) * np.log(expected_pct / actual_pct)

        return float(psi)

    def get_psi_values(self) -> dict[str, float]:
        'Get PSI values for all features.'
        return self.psi_values_.copy()

    def get_params(self) -> dict:
        'Get filter parameters.'
        return {
            'threshold': self.threshold,
            'n_bins': self.n_bins,
            'psi_values': self.psi_values_,
            'selected_features': self.selected_features_,
            'removed_features': self.removed_features_
        }

    def set_params(self, params: dict) -> 'PSIFilter':
        'Set filter parameters.'
        self.threshold = params.get('threshold', 0.25)
        self.n_bins = params.get('n_bins', 10)
        self.psi_values_ = params.get('psi_values', {})
        self.selected_features_ = params.get('selected_features', [])
        self.removed_features_ = params.get('removed_features', [])
        self.is_fitted_ = True
        return self
