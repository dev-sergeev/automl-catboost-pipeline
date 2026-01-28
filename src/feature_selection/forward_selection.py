'Accelerated forward feature selection.'

from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from src.feature_selection.base import BaseSelector
from src.utils import get_logger


class ForwardSelection(BaseSelector):
    '''
    Accelerated forward feature selection.

    Instead of adding one feature at a time, adds a percentage of
    the best features per iteration based on feature importance.

    Speed: 5 (slowest)
    '''

    def __init__(
        self,
        add_ratio: float = 0.10,
        max_features: int = 100,
        max_iterations: int = 20,
        metric_threshold: float = 0.001,
        random_seed: int = 42,
        cat_features: Optional[list[str]] = None
    ):
        super().__init__()
        self.add_ratio = add_ratio
        self.max_features = max_features
        self.max_iterations = max_iterations
        self.metric_threshold = metric_threshold
        self.random_seed = random_seed
        self.cat_features = cat_features
        self.logger = get_logger('feature_selection.forward')
        self.iteration_history_: list[dict] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        initial_features: Optional[list[str]] = None
    ) -> 'ForwardSelection':
        '''
        Perform forward selection.

        Args:
            X: Training features
            y: Training target
            sample_weight: Sample weights for CatBoost
            X_valid: Validation features
            y_valid: Validation target
            initial_features: Starting feature set (if pre-computed importance)
        '''
        self.logger.info(
            f'Starting ForwardSelection (add_ratio={self.add_ratio}, '
            f'max_features={self.max_features}) on {X.shape[1]} features...'
        )

        if X_valid is None or y_valid is None:
            n_valid = int(len(X) * 0.2)
            X_valid = X.iloc[-n_valid:]
            y_valid = y.iloc[-n_valid:]
            X = X.iloc[:-n_valid]
            y = y.iloc[:-n_valid]
            if sample_weight is not None:
                sample_weight = sample_weight[:-n_valid]

        all_features = list(X.columns)

        # Get initial feature ranking using feature importance
        if initial_features is None:
            initial_features = self._get_initial_ranking(
                X, y, sample_weight, all_features
            )

        # Start with top features
        n_initial = max(5, int(len(all_features) * self.add_ratio))
        current_features = initial_features[:n_initial]
        remaining_features = initial_features[n_initial:]

        best_score = 0.0
        best_features = current_features.copy()
        self.iteration_history_ = []

        for iteration in range(self.max_iterations):
            if len(current_features) >= self.max_features:
                self.logger.info(
                    f'Reached max features ({self.max_features}), stopping'
                )
                break

            if not remaining_features:
                self.logger.info('No more features to add, stopping')
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

            # Check if improvement is too small
            if iteration > 0:
                prev_score = self.iteration_history_[-2]['score']
                if score - prev_score < self.metric_threshold:
                    self.logger.info(
                        f'Improvement too small ({score - prev_score:.4f}), stopping'
                    )
                    break

            # Add more features
            n_to_add = max(1, int(len(all_features) * self.add_ratio))
            features_to_add = remaining_features[:n_to_add]
            remaining_features = remaining_features[n_to_add:]
            current_features = current_features + features_to_add

        self.selected_features_ = best_features
        self.removed_features_ = [
            f for f in all_features if f not in best_features
        ]

        self.logger.info(
            f'ForwardSelection complete: {len(self.selected_features_)} features, '
            f'best AUC={best_score:.4f}'
        )

        self._params = {
            'add_ratio': self.add_ratio,
            'max_features': self.max_features,
            'iteration_history': self.iteration_history_
        }
        self.is_fitted_ = True
        return self

    def _get_initial_ranking(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray],
        features: list[str]
    ) -> list[str]:
        'Get initial feature ranking using quick model.'
        self.logger.info('Computing initial feature ranking...')

        cat_features = self._get_cat_features(X)

        model = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=self.random_seed,
            verbose=0,
            thread_count=-1,
            cat_features=cat_features
        )

        model.fit(X, y, sample_weight=sample_weight, verbose=0)

        importance = dict(zip(features, model.feature_importances_))
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        return [f for f, _ in sorted_features]

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
            'add_ratio': self.add_ratio,
            'max_features': self.max_features,
            'max_iterations': self.max_iterations,
            'metric_threshold': self.metric_threshold,
            'iteration_history': self.iteration_history_,
            'selected_features': self.selected_features_,
            'removed_features': self.removed_features_
        }

    def set_params(self, params: dict) -> 'ForwardSelection':
        'Set selector parameters.'
        self.add_ratio = params.get('add_ratio', 0.10)
        self.max_features = params.get('max_features', 100)
        self.max_iterations = params.get('max_iterations', 20)
        self.metric_threshold = params.get('metric_threshold', 0.001)
        self.iteration_history_ = params.get('iteration_history', [])
        self.selected_features_ = params.get('selected_features', [])
        self.removed_features_ = params.get('removed_features', [])
        self.is_fitted_ = True
        return self
