'Main feature selector orchestrating all selection steps.'

from typing import Optional

import numpy as np
import pandas as pd

from src.config import FeatureSelectionConfig, PipelineConfig
from src.feature_selection.backward_selection import BackwardSelection
from src.feature_selection.correlation_filter import CorrelationFilter
from src.feature_selection.forward_selection import ForwardSelection
from src.feature_selection.importance_filter import ImportanceFilter
from src.feature_selection.missing_filter import MissingFilter
from src.feature_selection.psi_filter import PSIFilter
from src.feature_selection.psi_time_filter import PSITimeFilter
from src.feature_selection.variance_filter import VarianceFilter
from src.utils import NoFeaturesRemainingError, get_logger


class FeatureSelector:
    '''
    Main feature selector orchestrating all selection steps.

    Applies filters in order from fastest to slowest:
    1. MissingFilter (Speed=1)
    2. VarianceFilter (Speed=1)
    3. CorrelationFilter (Speed=2)
    4. PSIFilter (Speed=2) - train vs valid comparison
    5. PSITimeFilter (Speed=2) - stability over time periods
    6. ImportanceFilter (Speed=3)
    7. BackwardSelection (Speed=4, optional)
    8. ForwardSelection (Speed=5, optional)
    '''

    def __init__(
        self,
        config: PipelineConfig,
        selection_config: Optional[FeatureSelectionConfig] = None
    ):
        self.config = config
        self.selection_config = selection_config or FeatureSelectionConfig()
        self.logger = get_logger('feature_selection')

        self.filters_: dict = {}
        self.selected_features_: list[str] = []
        self.selection_history_: list[dict] = []
        self.is_fitted_: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        cat_features: Optional[list[str]] = None,
        date_column: Optional[str] = None,
        train_dates: Optional[pd.Series] = None,
        valid_dates: Optional[pd.Series] = None
    ) -> 'FeatureSelector':
        '''
        Fit all feature selection steps.

        Args:
            X_train: Training features
            y_train: Training target
            X_valid: Validation features
            y_valid: Validation target
            sample_weight: Sample weights for training
            cat_features: Categorical feature names
            date_column: Name of date column (for PSITimeFilter)
            train_dates: Date values for train set (for PSITimeFilter)
            valid_dates: Date values for valid set (for PSITimeFilter)
        '''
        self.logger.info(
            f'Starting feature selection on {X_train.shape[1]} features...'
        )

        self.selection_history_ = []
        current_features = list(X_train.columns)

        # Get thresholds
        sc = self.selection_config
        cfg = self.config

        # 1. Missing Filter
        if sc.run_missing_filter:
            threshold = sc.missing_threshold or cfg.missing_threshold
            filter_ = MissingFilter(threshold=threshold)
            filter_.fit(X_train[current_features], y_train)
            current_features = filter_.get_selected_features()
            self.filters_['missing'] = filter_
            self._log_step('MissingFilter', filter_, current_features)
            self._check_features_remaining(current_features, 'MissingFilter')

        # 2. Variance Filter
        if sc.run_variance_filter:
            threshold = sc.variance_threshold or cfg.variance_threshold
            filter_ = VarianceFilter(threshold=threshold)
            filter_.fit(X_train[current_features], y_train)
            current_features = filter_.get_selected_features()
            self.filters_['variance'] = filter_
            self._log_step('VarianceFilter', filter_, current_features)
            self._check_features_remaining(current_features, 'VarianceFilter')

        # 3. Correlation Filter
        if sc.run_correlation_filter:
            threshold = sc.correlation_threshold or cfg.correlation_threshold
            filter_ = CorrelationFilter(threshold=threshold)
            filter_.fit(X_train[current_features], y_train)
            current_features = filter_.get_selected_features()
            self.filters_['correlation'] = filter_
            self._log_step('CorrelationFilter', filter_, current_features)
            self._check_features_remaining(current_features, 'CorrelationFilter')

        # 4. PSI Filter (train vs valid)
        if sc.run_psi_filter:
            threshold = sc.psi_threshold or cfg.psi_threshold
            filter_ = PSIFilter(threshold=threshold)
            filter_.fit(
                X_valid[current_features],
                y_valid,
                X_reference=X_train[current_features]
            )
            current_features = filter_.get_selected_features()
            self.filters_['psi'] = filter_
            self._log_step('PSIFilter', filter_, current_features)
            self._check_features_remaining(current_features, 'PSIFilter')

        # 5. PSI Time Filter (stability over time)
        if sc.run_psi_time_filter and date_column and train_dates is not None:
            threshold = sc.psi_threshold or cfg.psi_threshold
            period_days = sc.psi_time_period_days
            min_period_days = sc.psi_time_min_period_days

            # Combine train and valid with dates
            X_combined = pd.concat([
                X_train[current_features].assign(**{date_column: train_dates.values}),
                X_valid[current_features].assign(**{date_column: valid_dates.values})
            ], ignore_index=True)
            y_combined = pd.concat([y_train, y_valid], ignore_index=True)

            filter_ = PSITimeFilter(
                date_column=date_column,
                threshold=threshold,
                period_days=period_days,
                min_period_days=min_period_days
            )
            filter_.fit(X_combined, y_combined)

            # Filter current features to only those selected by PSITimeFilter
            current_features = [
                f for f in current_features
                if f in filter_.get_selected_features()
            ]
            self.filters_['psi_time'] = filter_
            self._log_step('PSITimeFilter', filter_, current_features)
            self._check_features_remaining(current_features, 'PSITimeFilter')

        # 6. Importance Filter
        if sc.run_importance_filter:
            threshold = sc.importance_threshold or cfg.importance_threshold
            filter_ = ImportanceFilter(
                threshold=threshold,
                random_seed=cfg.random_seed,
                cat_features=cat_features
            )
            filter_.fit(
                X_train[current_features],
                y_train,
                sample_weight=sample_weight,
                X_valid=X_valid[current_features],
                y_valid=y_valid
            )
            current_features = filter_.get_selected_features()
            self.filters_['importance'] = filter_
            self._log_step('ImportanceFilter', filter_, current_features)
            self._check_features_remaining(current_features, 'ImportanceFilter')

        # 7. Backward Selection (optional)
        if sc.run_backward_selection:
            drop_ratio = sc.backward_drop_ratio or cfg.backward_drop_ratio
            filter_ = BackwardSelection(
                drop_ratio=drop_ratio,
                min_features=sc.backward_min_features,
                random_seed=cfg.random_seed,
                cat_features=cat_features
            )
            filter_.fit(
                X_train[current_features],
                y_train,
                sample_weight=sample_weight,
                X_valid=X_valid[current_features],
                y_valid=y_valid
            )
            current_features = filter_.get_selected_features()
            self.filters_['backward'] = filter_
            self._log_step('BackwardSelection', filter_, current_features)
            self._check_features_remaining(current_features, 'BackwardSelection')

        # 8. Forward Selection (optional)
        if sc.run_forward_selection:
            add_ratio = sc.forward_add_ratio or cfg.forward_add_ratio
            filter_ = ForwardSelection(
                add_ratio=add_ratio,
                max_features=sc.forward_max_features,
                random_seed=cfg.random_seed,
                cat_features=cat_features
            )
            filter_.fit(
                X_train[current_features],
                y_train,
                sample_weight=sample_weight,
                X_valid=X_valid[current_features],
                y_valid=y_valid
            )
            current_features = filter_.get_selected_features()
            self.filters_['forward'] = filter_
            self._log_step('ForwardSelection', filter_, current_features)
            self._check_features_remaining(current_features, 'ForwardSelection')

        self.selected_features_ = current_features
        self.is_fitted_ = True

        self.logger.info(
            f'Feature selection complete: '
            f'{X_train.shape[1]} -> {len(self.selected_features_)} features'
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        'Select fitted features from DataFrame.'
        if not self.is_fitted_:
            raise RuntimeError('FeatureSelector must be fitted before transform')

        available = [f for f in self.selected_features_ if f in X.columns]
        return X[available]

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        cat_features: Optional[list[str]] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        'Fit and transform in one step.'
        self.fit(
            X_train, y_train, X_valid, y_valid,
            sample_weight=sample_weight,
            cat_features=cat_features
        )
        return self.transform(X_train), self.transform(X_valid)

    def _log_step(
        self,
        name: str,
        filter_,
        remaining_features: list[str]
    ) -> None:
        'Log feature selection step.'
        n_removed = len(filter_.get_removed_features())
        n_remaining = len(remaining_features)

        self.selection_history_.append({
            'step': name,
            'removed': n_removed,
            'remaining': n_remaining,
            'removed_features': filter_.get_removed_features()
        })

        self.logger.info(
            f'{name}: removed {n_removed} features, '
            f'{n_remaining} remaining'
        )

    def _check_features_remaining(
        self,
        features: list[str],
        filter_name: str
    ) -> None:
        '''
        Check if any features remain after a filter step.

        Raises NoFeaturesRemainingError if all features were removed.
        '''
        if len(features) == 0:
            message = (
                f'All features were removed during feature selection. '
                f'Last filter: {filter_name}. '
                f'See selection_history_ for details.'
            )
            raise NoFeaturesRemainingError(
                message=message,
                selection_history=self.selection_history_,
                last_filter=filter_name
            )

    def get_selected_features(self) -> list[str]:
        'Get list of selected features.'
        return self.selected_features_.copy()

    def get_selection_history(self) -> list[dict]:
        'Get history of all selection steps.'
        return self.selection_history_.copy()

    def get_filter(self, name: str):
        'Get a specific filter by name.'
        return self.filters_.get(name)

    def get_params(self) -> dict:
        'Get selector parameters for serialization.'
        filter_params = {
            name: f.get_params() for name, f in self.filters_.items()
        }
        return {
            'selected_features': self.selected_features_,
            'selection_history': self.selection_history_,
            'filter_params': filter_params
        }

    def set_params(self, params: dict) -> 'FeatureSelector':
        'Set selector parameters from serialization.'
        self.selected_features_ = params.get('selected_features', [])
        self.selection_history_ = params.get('selection_history', [])
        self.is_fitted_ = True
        return self
