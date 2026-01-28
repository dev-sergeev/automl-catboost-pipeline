'Accelerated backward feature selection.'

from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from src.feature_selection.base import BaseSelector
from src.utils import get_logger


class BackwardSelection(BaseSelector):
    '''
    Accelerated backward feature selection.

    Instead of removing one feature at a time, removes a percentage
    of the worst features per iteration based on feature importance.

    Speed: 4 (slow)
    '''

    def __init__(
        self,
        drop_ratio: float = 0.10,
        min_features: int = 10,
        max_iterations: int = 20,
        metric_threshold: float = 0.01,
        random_seed: int = 42,
        cat_features: Optional[list[str]] = None
    ):
        super().__init__()
        self.drop_ratio = drop_ratio
        self.min_features = min_features
        self.max_iterations = max_iterations
        self.metric_threshold = metric_threshold
        self.random_seed = random_seed
        self.cat_features = cat_features
        self.logger = get_logger('feature_selection.backward')
        self.iteration_history_: list[dict] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None
    ) -> 'BackwardSelection':
        '''
        Perform backward selection.

        Args:
            X: Training features
            y: Training target
            sample_weight: Sample weights for CatBoost
            X_valid: Validation features (for metric computation)
            y_valid: Validation target
        '''
        self.logger.info(
            f'Starting BackwardSelection (drop_ratio={self.drop_ratio}, '
            f'min_features={self.min_features}) on {X.shape[1]} features...'
        )

        if X_valid is None or y_valid is None:
            # Split data
            n_valid = int(len(X) * 0.2)
            X_valid = X.iloc[-n_valid:]
            y_valid = y.iloc[-n_valid:]
            X = X.iloc[:-n_valid]
            y = y.iloc[:-n_valid]
            if sample_weight is not None:
                sample_weight = sample_weight[:-n_valid]

        current_features = list(X.columns)
        best_score = 0.0
        best_features = current_features.copy()
        self.iteration_history_ = []

        for iteration in range(self.max_iterations):
            if len(current_features) <= self.min_features:
                self.logger.info(
                    f'Reached minimum features ({self.min_features}), stopping'
                )
                break

            # Train model with current features
            X_train_iter = X[current_features]
            X_valid_iter = X_valid[current_features]
            cat_features = self._get_cat_features(X_train_iter)

            model = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_seed=self.random_seed,
                verbose=0,
                thread_count=-1,
                cat_features=cat_features
            )

            model.fit(X_train_iter, y, sample_weight=sample_weight, verbose=0)

            # Evaluate
            y_pred = model.predict_proba(X_valid_iter)[:, 1]
            score = roc_auc_score(y_valid, y_pred)

            # Get feature importance
            importance = dict(zip(current_features, model.feature_importances_))

            # Record iteration
            self.iteration_history_.append({
                'iteration': iteration,
                'n_features': len(current_features),
                'score': score,
                'features': current_features.copy()
            })

            self.logger.info(
                f'Iteration {iteration}: {len(current_features)} features, '
                f'AUC={score:.4f}'
            )

            # Update best
            if score > best_score:
                best_score = score
                best_features = current_features.copy()

            # Check if score dropped significantly
            if best_score - score > self.metric_threshold:
                self.logger.info(
                    f'Score dropped by {best_score - score:.4f}, stopping'
                )
                break

            # Remove worst features
            n_to_drop = max(1, int(len(current_features) * self.drop_ratio))

            # Sort by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1])
            features_to_remove = [f for f, _ in sorted_features[:n_to_drop]]

            current_features = [
                f for f in current_features if f not in features_to_remove
            ]

        self.selected_features_ = best_features
        self.removed_features_ = [
            f for f in X.columns if f not in best_features
        ]

        self.logger.info(
            f'BackwardSelection complete: {len(self.selected_features_)} features, '
            f'best AUC={best_score:.4f}'
        )

        self._params = {
            'drop_ratio': self.drop_ratio,
            'min_features': self.min_features,
            'iteration_history': self.iteration_history_
        }
        self.is_fitted_ = True
        return self

    def _get_cat_features(self, X: pd.DataFrame) -> list[int]:
        'Get categorical feature indices.'
        if self.cat_features:
            return [
                i for i, col in enumerate(X.columns)
                if col in self.cat_features
            ]

        return [
            i for i, col in enumerate(X.columns)
            if not pd.api.types.is_numeric_dtype(X[col].dtype)
        ]

    def get_iteration_history(self) -> list[dict]:
        'Get history of all iterations.'
        return self.iteration_history_.copy()

    def get_params(self) -> dict:
        'Get selector parameters.'
        return {
            'drop_ratio': self.drop_ratio,
            'min_features': self.min_features,
            'max_iterations': self.max_iterations,
            'metric_threshold': self.metric_threshold,
            'iteration_history': self.iteration_history_,
            'selected_features': self.selected_features_,
            'removed_features': self.removed_features_
        }

    def set_params(self, params: dict) -> 'BackwardSelection':
        'Set selector parameters.'
        self.drop_ratio = params.get('drop_ratio', 0.10)
        self.min_features = params.get('min_features', 10)
        self.max_iterations = params.get('max_iterations', 20)
        self.metric_threshold = params.get('metric_threshold', 0.01)
        self.iteration_history_ = params.get('iteration_history', [])
        self.selected_features_ = params.get('selected_features', [])
        self.removed_features_ = params.get('removed_features', [])
        self.is_fitted_ = True
        return self
