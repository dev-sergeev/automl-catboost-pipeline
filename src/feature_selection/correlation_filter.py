'Filter highly correlated features.'

from typing import Optional

import numpy as np
import pandas as pd

from src.feature_selection.base import BaseSelector
from src.utils import get_logger


class CorrelationFilter(BaseSelector):
    '''
    Remove one feature from highly correlated pairs.

    For each pair with |correlation| > threshold, removes the feature
    with lower correlation to target.

    Only applies to numeric features.

    Speed: 2 (medium-fast)
    '''

    def __init__(self, threshold: float = 0.95):
        super().__init__()
        self.threshold = threshold
        self.logger = get_logger('feature_selection.correlation')
        self.correlation_pairs_: list[tuple[str, str, float]] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'CorrelationFilter':
        'Identify and remove highly correlated features.'
        self.logger.info(
            f'Fitting CorrelationFilter (threshold={self.threshold}) '
            f'on {X.shape[1]} features...'
        )

        # Get numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in X.columns if c not in numeric_cols]

        if len(numeric_cols) < 2:
            self.selected_features_ = list(X.columns)
            self.removed_features_ = []
            self.is_fitted_ = True
            return self

        # Compute correlation matrix
        corr_matrix = X[numeric_cols].corr().abs()

        # Compute correlation with target for tie-breaking
        target_corr = {}
        for col in numeric_cols:
            target_corr[col] = abs(X[col].corr(y))

        # Find highly correlated pairs
        self.correlation_pairs_ = []
        to_remove = set()

        # Get upper triangle indices
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

        for i, col1 in enumerate(numeric_cols):
            if col1 in to_remove:
                continue

            for j, col2 in enumerate(numeric_cols[i + 1:], start=i + 1):
                if col2 in to_remove:
                    continue

                if upper_tri[i, j] and corr_matrix.iloc[i, j] > self.threshold:
                    corr_value = corr_matrix.iloc[i, j]
                    self.correlation_pairs_.append((col1, col2, corr_value))

                    # Remove feature with lower target correlation
                    if target_corr.get(col1, 0) >= target_corr.get(col2, 0):
                        to_remove.add(col2)
                    else:
                        to_remove.add(col1)

        self.removed_features_ = list(to_remove)
        self.selected_features_ = [
            c for c in numeric_cols if c not in to_remove
        ] + non_numeric_cols

        self.logger.info(
            f'Found {len(self.correlation_pairs_)} highly correlated pairs, '
            f'removed {len(self.removed_features_)} features'
        )

        self._params = {
            'threshold': self.threshold,
            'correlation_pairs': self.correlation_pairs_
        }
        self.is_fitted_ = True
        return self

    def get_correlation_pairs(self) -> list[tuple[str, str, float]]:
        'Get list of highly correlated pairs.'
        return self.correlation_pairs_.copy()

    def get_params(self) -> dict:
        'Get filter parameters.'
        return {
            'threshold': self.threshold,
            'correlation_pairs': self.correlation_pairs_,
            'selected_features': self.selected_features_,
            'removed_features': self.removed_features_
        }

    def set_params(self, params: dict) -> 'CorrelationFilter':
        'Set filter parameters.'
        self.threshold = params.get('threshold', 0.95)
        self.correlation_pairs_ = params.get('correlation_pairs', [])
        self.selected_features_ = params.get('selected_features', [])
        self.removed_features_ = params.get('removed_features', [])
        self.is_fitted_ = True
        return self
