'Tests for Optuna optimizer.'

import numpy as np
import pandas as pd
import pytest

from src.config import OptunaConfig, PipelineConfig
from src.optimization import OptunaOptimizer


class TestOptunaOptimizer:
    'Tests for OptunaOptimizer.'

    @pytest.fixture
    def quick_config(self, random_seed):
        'Configuration for quick optimization.'
        return PipelineConfig(
            id_columns='client_id',
            random_seed=random_seed,
            n_trials=3,  # Very few trials for testing
            optuna_timeout=30,
            catboost_iterations=20  # Few iterations for testing
        )

    @pytest.fixture
    def quick_optuna_config(self):
        'Quick Optuna config for testing.'
        return OptunaConfig(
            n_trials=3,
            timeout=30,
            use_pruning=False,
            show_progress_bar=False
        )

    def test_init(self, quick_config, quick_optuna_config):
        'Test optimizer initialization.'
        optimizer = OptunaOptimizer(quick_config, quick_optuna_config)

        assert optimizer.config == quick_config
        assert optimizer.optuna_config == quick_optuna_config
        assert optimizer.is_fitted_ is False

    def test_optimize_basic(self, quick_config, quick_optuna_config, sample_train_valid_data):
        'Test basic optimization.'
        X_train, y_train, X_valid, y_valid = sample_train_valid_data

        # Use only numeric columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train = X_train[numeric_cols]
        X_valid = X_valid[numeric_cols]

        optimizer = OptunaOptimizer(quick_config, quick_optuna_config)
        best_params = optimizer.optimize(X_train, y_train, X_valid, y_valid)

        assert optimizer.is_fitted_
        assert isinstance(best_params, dict)
        assert 'iterations' in best_params or 'depth' in best_params

    def test_get_best_params(self, quick_config, quick_optuna_config, sample_train_valid_data):
        'Test getting best parameters.'
        X_train, y_train, X_valid, y_valid = sample_train_valid_data

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train = X_train[numeric_cols]
        X_valid = X_valid[numeric_cols]

        optimizer = OptunaOptimizer(quick_config, quick_optuna_config)
        optimizer.optimize(X_train, y_train, X_valid, y_valid)

        params = optimizer.get_best_params()

        assert isinstance(params, dict)

    def test_get_best_score(self, quick_config, quick_optuna_config, sample_train_valid_data):
        'Test getting best score.'
        X_train, y_train, X_valid, y_valid = sample_train_valid_data

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train = X_train[numeric_cols]
        X_valid = X_valid[numeric_cols]

        optimizer = OptunaOptimizer(quick_config, quick_optuna_config)
        optimizer.optimize(X_train, y_train, X_valid, y_valid)

        score = optimizer.get_best_score()

        assert isinstance(score, float)
        assert 0 <= score <= 1  # AUC should be between 0 and 1

    def test_get_trials_dataframe(self, quick_config, quick_optuna_config, sample_train_valid_data):
        'Test getting trials as DataFrame.'
        X_train, y_train, X_valid, y_valid = sample_train_valid_data

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train = X_train[numeric_cols]
        X_valid = X_valid[numeric_cols]

        optimizer = OptunaOptimizer(quick_config, quick_optuna_config)
        optimizer.optimize(X_train, y_train, X_valid, y_valid)

        trials_df = optimizer.get_trials_dataframe()

        assert isinstance(trials_df, pd.DataFrame)
        assert len(trials_df) == 3  # Number of trials

    def test_get_study(self, quick_config, quick_optuna_config, sample_train_valid_data):
        'Test getting Optuna study.'
        X_train, y_train, X_valid, y_valid = sample_train_valid_data

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train = X_train[numeric_cols]
        X_valid = X_valid[numeric_cols]

        optimizer = OptunaOptimizer(quick_config, quick_optuna_config)
        optimizer.optimize(X_train, y_train, X_valid, y_valid)

        study = optimizer.get_study()

        assert study is not None
        assert len(study.trials) == 3

    def test_with_sample_weight(self, quick_config, quick_optuna_config, sample_train_valid_data):
        'Test optimization with sample weights.'
        X_train, y_train, X_valid, y_valid = sample_train_valid_data

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train = X_train[numeric_cols]
        X_valid = X_valid[numeric_cols]

        sample_weight = np.random.rand(len(X_train))

        optimizer = OptunaOptimizer(quick_config, quick_optuna_config)
        best_params = optimizer.optimize(
            X_train, y_train, X_valid, y_valid,
            sample_weight=sample_weight
        )

        assert optimizer.is_fitted_
        assert isinstance(best_params, dict)

    def test_with_cat_features(self, quick_config, quick_optuna_config, sample_train_valid_data):
        'Test optimization with categorical features.'
        X_train, y_train, X_valid, y_valid = sample_train_valid_data

        # Include categorical column
        cat_cols = X_train.select_dtypes(include=['object']).columns
        if len(cat_cols) == 0:
            X_train['cat'] = np.random.choice(['A', 'B'], len(X_train))
            X_valid['cat'] = np.random.choice(['A', 'B'], len(X_valid))
            cat_features = [X_train.columns.get_loc('cat')]
        else:
            cat_features = [X_train.columns.get_loc(c) for c in cat_cols]

        optimizer = OptunaOptimizer(quick_config, quick_optuna_config)
        optimizer.optimize(
            X_train, y_train, X_valid, y_valid,
            cat_features=cat_features
        )

        assert optimizer.is_fitted_

    def test_get_set_params(self, quick_config, quick_optuna_config, sample_train_valid_data):
        'Test serialization.'
        X_train, y_train, X_valid, y_valid = sample_train_valid_data

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train = X_train[numeric_cols]
        X_valid = X_valid[numeric_cols]

        optimizer1 = OptunaOptimizer(quick_config, quick_optuna_config)
        optimizer1.optimize(X_train, y_train, X_valid, y_valid)
        params = optimizer1.get_params()

        optimizer2 = OptunaOptimizer(quick_config, quick_optuna_config)
        optimizer2.set_params(params)

        assert optimizer2.best_params_ == optimizer1.best_params_
        assert optimizer2.best_score_ == optimizer1.best_score_
        assert optimizer2.is_fitted_

    def test_not_fitted_raises(self, quick_config, quick_optuna_config):
        'Test that methods raise before fitting.'
        optimizer = OptunaOptimizer(quick_config, quick_optuna_config)

        with pytest.raises(RuntimeError):
            optimizer.get_best_params()

        with pytest.raises(RuntimeError):
            optimizer.get_best_score()

        with pytest.raises(RuntimeError):
            optimizer.get_study()
