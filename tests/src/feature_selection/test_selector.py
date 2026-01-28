'Tests for feature selector orchestrator.'

import numpy as np
import pandas as pd
import pytest

from src.config import FeatureSelectionConfig
from src.feature_selection import FeatureSelector


class TestFeatureSelector:
    'Tests for FeatureSelector.'

    @pytest.fixture
    def sample_features_df(self):
        'Create sample feature DataFrame.'
        np.random.seed(42)
        n = 500

        return pd.DataFrame({
            'good_numeric': np.random.randn(n),
            'good_numeric_2': np.random.randn(n) * 10,
            'high_missing': np.where(np.random.random(n) > 0.1, np.nan, np.random.randn(n)),
            'zero_variance': np.ones(n) * 5,
            'correlated': np.random.randn(n),
            'categorical': np.random.choice(['A', 'B', 'C'], n),
        })

    @pytest.fixture
    def sample_target(self, sample_features_df):
        'Create sample target.'
        n = len(sample_features_df)
        return pd.Series(np.random.randint(0, 2, n))

    def test_init(self, sample_config, sample_selection_config):
        'Test selector initialization.'
        selector = FeatureSelector(sample_config, sample_selection_config)

        assert selector.config == sample_config
        assert selector.selection_config == sample_selection_config

    def test_fit_basic(self, sample_features_df, sample_target, sample_config, sample_selection_config):
        'Test basic fit.'
        # Split data
        n = len(sample_features_df)
        train_idx = int(n * 0.7)

        X_train = sample_features_df.iloc[:train_idx]
        y_train = sample_target.iloc[:train_idx]
        X_valid = sample_features_df.iloc[train_idx:]
        y_valid = sample_target.iloc[train_idx:]

        selector = FeatureSelector(sample_config, sample_selection_config)
        selector.fit(X_train, y_train, X_valid, y_valid)

        assert selector.is_fitted_
        assert len(selector.selected_features_) > 0

    def test_removes_high_missing(self, sample_features_df, sample_target, sample_config):
        'Test that high missing features are removed.'
        selection_config = FeatureSelectionConfig(
            run_missing_filter=True,
            run_variance_filter=False,
            run_correlation_filter=False,
            run_psi_filter=False,
            run_importance_filter=False,
            missing_threshold=0.5
        )

        n = len(sample_features_df)
        train_idx = int(n * 0.7)

        X_train = sample_features_df.iloc[:train_idx]
        y_train = sample_target.iloc[:train_idx]
        X_valid = sample_features_df.iloc[train_idx:]
        y_valid = sample_target.iloc[train_idx:]

        selector = FeatureSelector(sample_config, selection_config)
        selector.fit(X_train, y_train, X_valid, y_valid)

        assert 'high_missing' not in selector.selected_features_

    def test_removes_zero_variance(self, sample_features_df, sample_target, sample_config):
        'Test that zero variance features are removed.'
        selection_config = FeatureSelectionConfig(
            run_missing_filter=False,
            run_variance_filter=True,
            run_correlation_filter=False,
            run_psi_filter=False,
            run_importance_filter=False,
        )

        n = len(sample_features_df)
        train_idx = int(n * 0.7)

        X_train = sample_features_df.iloc[:train_idx]
        y_train = sample_target.iloc[:train_idx]
        X_valid = sample_features_df.iloc[train_idx:]
        y_valid = sample_target.iloc[train_idx:]

        selector = FeatureSelector(sample_config, selection_config)
        selector.fit(X_train, y_train, X_valid, y_valid)

        assert 'zero_variance' not in selector.selected_features_

    def test_transform(self, sample_features_df, sample_target, sample_config, sample_selection_config):
        'Test transform returns selected features.'
        n = len(sample_features_df)
        train_idx = int(n * 0.7)

        X_train = sample_features_df.iloc[:train_idx]
        y_train = sample_target.iloc[:train_idx]
        X_valid = sample_features_df.iloc[train_idx:]
        y_valid = sample_target.iloc[train_idx:]

        selector = FeatureSelector(sample_config, sample_selection_config)
        selector.fit(X_train, y_train, X_valid, y_valid)

        result = selector.transform(X_train)

        assert set(result.columns) == set(selector.selected_features_)

    def test_fit_transform(self, sample_features_df, sample_target, sample_config, sample_selection_config):
        'Test fit_transform.'
        n = len(sample_features_df)
        train_idx = int(n * 0.7)

        X_train = sample_features_df.iloc[:train_idx]
        y_train = sample_target.iloc[:train_idx]
        X_valid = sample_features_df.iloc[train_idx:]
        y_valid = sample_target.iloc[train_idx:]

        selector = FeatureSelector(sample_config, sample_selection_config)
        X_train_selected, X_valid_selected = selector.fit_transform(
            X_train, y_train, X_valid, y_valid
        )

        assert set(X_train_selected.columns) == set(selector.selected_features_)
        assert set(X_valid_selected.columns) == set(selector.selected_features_)

    def test_get_selection_history(self, sample_features_df, sample_target, sample_config, sample_selection_config):
        'Test getting selection history.'
        n = len(sample_features_df)
        train_idx = int(n * 0.7)

        X_train = sample_features_df.iloc[:train_idx]
        y_train = sample_target.iloc[:train_idx]
        X_valid = sample_features_df.iloc[train_idx:]
        y_valid = sample_target.iloc[train_idx:]

        selector = FeatureSelector(sample_config, sample_selection_config)
        selector.fit(X_train, y_train, X_valid, y_valid)

        history = selector.get_selection_history()

        assert isinstance(history, list)
        assert len(history) > 0
        assert all('step' in h for h in history)
        assert all('removed' in h for h in history)
        assert all('remaining' in h for h in history)

    def test_get_filter(self, sample_features_df, sample_target, sample_config, sample_selection_config):
        'Test getting specific filter.'
        n = len(sample_features_df)
        train_idx = int(n * 0.7)

        X_train = sample_features_df.iloc[:train_idx]
        y_train = sample_target.iloc[:train_idx]
        X_valid = sample_features_df.iloc[train_idx:]
        y_valid = sample_target.iloc[train_idx:]

        selector = FeatureSelector(sample_config, sample_selection_config)
        selector.fit(X_train, y_train, X_valid, y_valid)

        missing_filter = selector.get_filter('missing')
        assert missing_filter is not None

    def test_get_set_params(self, sample_features_df, sample_target, sample_config, sample_selection_config):
        'Test serialization.'
        n = len(sample_features_df)
        train_idx = int(n * 0.7)

        X_train = sample_features_df.iloc[:train_idx]
        y_train = sample_target.iloc[:train_idx]
        X_valid = sample_features_df.iloc[train_idx:]
        y_valid = sample_target.iloc[train_idx:]

        selector1 = FeatureSelector(sample_config, sample_selection_config)
        selector1.fit(X_train, y_train, X_valid, y_valid)
        params = selector1.get_params()

        selector2 = FeatureSelector(sample_config, sample_selection_config)
        selector2.set_params(params)

        assert selector2.selected_features_ == selector1.selected_features_
        assert selector2.is_fitted_

    def test_with_sample_weight(self, sample_features_df, sample_target, sample_config, sample_selection_config):
        'Test with sample weights.'
        n = len(sample_features_df)
        train_idx = int(n * 0.7)

        X_train = sample_features_df.iloc[:train_idx]
        y_train = sample_target.iloc[:train_idx]
        X_valid = sample_features_df.iloc[train_idx:]
        y_valid = sample_target.iloc[train_idx:]

        sample_weight = np.random.rand(len(X_train))

        selector = FeatureSelector(sample_config, sample_selection_config)
        selector.fit(X_train, y_train, X_valid, y_valid, sample_weight=sample_weight)

        assert selector.is_fitted_
        assert len(selector.selected_features_) > 0
