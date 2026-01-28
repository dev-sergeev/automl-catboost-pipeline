'Filter features with zero or near-zero variance.'

from typing import Optional

import numpy as np
import pandas as pd

from src.feature_selection.base import BaseSelector
from src.utils import get_logger


class VarianceFilter(BaseSelector):
    '''
    Remove features with variance below threshold.

    Features with zero variance (all same value) are uninformative.
    Only applies to numeric features.

    Speed: 1 (fastest)
    '''

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold
        self.logger = get_logger('feature_selection.variance')
        self.variances_: dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'VarianceFilter':
        'Identify features with low variance.'
        self.logger.info(
            f'Fitting VarianceFilter (threshold={self.threshold}) '
            f'on {X.shape[1]} features...'
        )

        self.selected_features_ = []
        self.removed_features_ = []
        self.variances_ = {}

        for col in X.columns:
            # Only compute variance for numeric columns
            if pd.api.types.is_numeric_dtype(X[col].dtype):
                variance = X[col].var()
                self.variances_[col] = variance if not pd.isna(variance) else 0.0

                if self.variances_[col] > self.threshold:
                    self.selected_features_.append(col)
                else:
                    self.removed_features_.append(col)
            else:
                # Keep non-numeric columns (categorical)
                # Check unique count for categorical
                n_unique = X[col].nunique()
                if n_unique > 1:
                    self.selected_features_.append(col)
                    self.variances_[col] = float(n_unique)  # Store unique count
                else:
                    self.removed_features_.append(col)
                    self.variances_[col] = 0.0

        self.logger.info(
            f'Removed {len(self.removed_features_)} features with '
            f'variance <= {self.threshold}'
        )

        self._params = {
            'threshold': self.threshold,
            'variances': self.variances_
        }
        self.is_fitted_ = True
        return self

    def get_variances(self) -> dict[str, float]:
        'Get variances for all features.'
        return self.variances_.copy()

    def get_params(self) -> dict:
        'Get filter parameters.'
        return {
            'threshold': self.threshold,
            'variances': self.variances_,
            'selected_features': self.selected_features_,
            'removed_features': self.removed_features_
        }

    def set_params(self, params: dict) -> 'VarianceFilter':
        'Set filter parameters.'
        self.threshold = params.get('threshold', 0.0)
        self.variances_ = params.get('variances', {})
        self.selected_features_ = params.get('selected_features', [])
        self.removed_features_ = params.get('removed_features', [])
        self.is_fitted_ = True
        return self
