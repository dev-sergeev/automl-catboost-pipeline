'Filter features based on CatBoost feature importance.'

from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.feature_selection.base import BaseSelector
from src.utils import get_logger


class ImportanceFilter(BaseSelector):
    '''
    Remove features with zero or low importance.

    Uses CatBoost's native feature importance (PredictionValuesChange)
    to identify and remove uninformative features.

    Speed: 3 (medium)
    '''

    def __init__(
        self,
        threshold: float = 0.0,
        importance_type: str = 'PredictionValuesChange',
        sample_size: Optional[int] = None,
        random_seed: int = 42,
        cat_features: Optional[list[str]] = None
    ):
        super().__init__()
        self.threshold = threshold
        self.importance_type = importance_type
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.cat_features = cat_features
        self.logger = get_logger('feature_selection.importance')
        self.importance_values_: dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None
    ) -> 'ImportanceFilter':
        '''
        Compute feature importance and filter features.

        Args:
            X: Training features
            y: Training target
            sample_weight: Sample weights for CatBoost
            X_valid: Validation features (for early stopping)
            y_valid: Validation target
        '''
        self.logger.info(
            f'Fitting ImportanceFilter (threshold={self.threshold}) '
            f'on {X.shape[1]} features...'
        )

        # Subsample if dataset is large
        X_train, y_train, weights = self._subsample(X, y, sample_weight)

        # Train a quick CatBoost model
        cat_features = self._get_cat_features(X_train)

        model = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=self.random_seed,
            verbose=0,
            thread_count=-1,
            cat_features=cat_features
        )

        # Use validation set for early stopping if provided
        eval_set = None
        if X_valid is not None and y_valid is not None:
            eval_set = (X_valid, y_valid)

        model.fit(
            X_train, y_train,
            sample_weight=weights,
            eval_set=eval_set,
            verbose=0
        )

        self.logger.info('Computing feature importance...')

        # Get CatBoost native feature importance
        importance = model.get_feature_importance(type=self.importance_type)

        # Normalize importance to [0, 1] range
        total_importance = importance.sum()
        if total_importance > 0:
            importance = importance / total_importance

        # Store importance values
        self.importance_values_ = {
            col: float(imp)
            for col, imp in zip(X.columns, importance)
        }

        # Filter features
        self.selected_features_ = []
        self.removed_features_ = []

        for col, imp in self.importance_values_.items():
            if imp > self.threshold:
                self.selected_features_.append(col)
            else:
                self.removed_features_.append(col)

        # Sort selected by importance
        self.selected_features_.sort(
            key=lambda x: self.importance_values_.get(x, 0),
            reverse=True
        )

        self.logger.info(
            f'Removed {len(self.removed_features_)} features with '
            f'importance <= {self.threshold}'
        )

        self._params = {
            'threshold': self.threshold,
            'importance_values': self.importance_values_
        }
        self.is_fitted_ = True
        return self

    def _subsample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray]
    ) -> tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]:
        'Subsample data if too large.'
        if self.sample_size and len(X) > self.sample_size:
            idx = np.random.RandomState(self.random_seed).choice(
                len(X), self.sample_size, replace=False
            )
            X = X.iloc[idx].reset_index(drop=True)
            y = y.iloc[idx].reset_index(drop=True)
            if sample_weight is not None:
                sample_weight = sample_weight[idx]
            self.logger.info(f'Subsampled to {self.sample_size} rows')

        return X, y, sample_weight

    def _get_cat_features(self, X: pd.DataFrame) -> list[int]:
        'Get categorical feature indices.'
        if self.cat_features:
            return [
                i for i, col in enumerate(X.columns)
                if col in self.cat_features
            ]

        # Auto-detect
        return [
            i for i, col in enumerate(X.columns)
            if not pd.api.types.is_numeric_dtype(X[col].dtype)
        ]

    def get_importance_values(self) -> dict[str, float]:
        'Get importance values for all features.'
        return self.importance_values_.copy()

    def get_top_features(self, n: int = 20) -> list[tuple[str, float]]:
        'Get top N features by importance.'
        sorted_features = sorted(
            self.importance_values_.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]

    def get_params(self) -> dict:
        'Get filter parameters.'
        return {
            'threshold': self.threshold,
            'importance_type': self.importance_type,
            'sample_size': self.sample_size,
            'importance_values': self.importance_values_,
            'selected_features': self.selected_features_,
            'removed_features': self.removed_features_
        }

    def set_params(self, params: dict) -> 'ImportanceFilter':
        'Set filter parameters.'
        self.threshold = params.get('threshold', 0.0)
        self.importance_type = params.get('importance_type', 'PredictionValuesChange')
        self.sample_size = params.get('sample_size')
        self.importance_values_ = params.get('importance_values', {})
        self.selected_features_ = params.get('selected_features', [])
        self.removed_features_ = params.get('removed_features', [])
        self.is_fitted_ = True
        return self
