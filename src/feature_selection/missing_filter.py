'Filter features with high missing rate.'

from typing import Optional

import numpy as np
import pandas as pd

from src.feature_selection.base import BaseSelector
from src.utils import get_logger


class MissingFilter(BaseSelector):
    '''
    Remove features with missing rate above threshold.

    Speed: 1 (fastest)
    '''

    def __init__(self, threshold: float = 0.95):
        super().__init__()
        self.threshold = threshold
        self.logger = get_logger('feature_selection.missing')
        self.missing_rates_: dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'MissingFilter':
        'Identify features with high missing rates.'
        self.logger.info(
            f'Fitting MissingFilter (threshold={self.threshold}) '
            f'on {X.shape[1]} features...'
        )

        self.selected_features_ = []
        self.removed_features_ = []
        self.missing_rates_ = {}

        for col in X.columns:
            missing_rate = X[col].isna().mean()
            self.missing_rates_[col] = missing_rate

            if missing_rate <= self.threshold:
                self.selected_features_.append(col)
            else:
                self.removed_features_.append(col)

        self.logger.info(
            f'Removed {len(self.removed_features_)} features with '
            f'missing rate > {self.threshold}'
        )

        self._params = {
            'threshold': self.threshold,
            'missing_rates': self.missing_rates_
        }
        self.is_fitted_ = True
        return self

    def get_missing_rates(self) -> dict[str, float]:
        'Get missing rates for all features.'
        return self.missing_rates_.copy()

    def get_params(self) -> dict:
        'Get filter parameters.'
        return {
            'threshold': self.threshold,
            'missing_rates': self.missing_rates_,
            'selected_features': self.selected_features_,
            'removed_features': self.removed_features_
        }

    def set_params(self, params: dict) -> 'MissingFilter':
        'Set filter parameters.'
        self.threshold = params.get('threshold', 0.95)
        self.missing_rates_ = params.get('missing_rates', {})
        self.selected_features_ = params.get('selected_features', [])
        self.removed_features_ = params.get('removed_features', [])
        self.is_fitted_ = True
        return self
