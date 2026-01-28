'Tests for main pipeline.'

import numpy as np
import pandas as pd
import pytest

from src.config import FeatureSelectionConfig, PipelineConfig
from src.pipeline import CatBoostPipeline
from src.scoring import Scorer


class TestCatBoostPipeline:
    'Tests for CatBoostPipeline.'

    @pytest.fixture
    def quick_config(self, random_seed, tmp_artifacts_dir):
        'Quick config for testing.'
        return PipelineConfig(
            id_columns=['client_id', 'app_id'],
            date_column='report_date',
            target_column='target',
            client_column='client_id',
            random_seed=random_seed,
            n_trials=2,  # Very few trials
            optuna_timeout=30,
            catboost_iterations=20,  # Few iterations
            catboost_early_stopping_rounds=5,
            artifacts_dir=tmp_artifacts_dir
        )

    @pytest.fixture
    def quick_selection_config(self):
        'Quick selection config.'
        return FeatureSelectionConfig(
            run_missing_filter=True,
            run_variance_filter=True,
            run_correlation_filter=False,  # Skip for speed
            run_psi_filter=False,
            run_importance_filter=False,
            run_backward_selection=False,
            run_forward_selection=False,
        )

    def test_init(self, quick_config, quick_selection_config):
        'Test pipeline initialization.'
        pipeline = CatBoostPipeline(quick_config, quick_selection_config)

        assert pipeline.config == quick_config
        assert pipeline.is_fitted_ is False

    def test_fit_basic(self, sample_df_small, quick_config, quick_selection_config):
        'Test basic pipeline fitting.'
        pipeline = CatBoostPipeline(quick_config, quick_selection_config)
        pipeline.fit(sample_df_small, run_optimization=False, save_artifacts=False)

        assert pipeline.is_fitted_
        assert pipeline.model_ is not None

    def test_fit_with_optimization(self, sample_df_small, quick_config, quick_selection_config):
        'Test pipeline with optimization.'
        pipeline = CatBoostPipeline(quick_config, quick_selection_config)
        pipeline.fit(sample_df_small, run_optimization=True, save_artifacts=False)

        assert pipeline.is_fitted_
        assert pipeline.optimizer.is_fitted_

    def test_predict(self, sample_df_small, quick_config, quick_selection_config):
        'Test prediction.'
        pipeline = CatBoostPipeline(quick_config, quick_selection_config)
        pipeline.fit(sample_df_small, run_optimization=False, save_artifacts=False)

        predictions = pipeline.predict(sample_df_small)

        assert len(predictions) == len(sample_df_small)
        assert all(0 <= p <= 1 for p in predictions)

    def test_get_metrics(self, sample_df_small, quick_config, quick_selection_config):
        'Test getting metrics.'
        pipeline = CatBoostPipeline(quick_config, quick_selection_config)
        pipeline.fit(sample_df_small, run_optimization=False, save_artifacts=False)

        metrics = pipeline.get_metrics()

        assert 'train' in metrics
        assert 'valid' in metrics
        assert 'oos' in metrics
        assert 'oot' in metrics
        assert 'gini' in metrics['train']
        assert 'auc' in metrics['train']

    def test_get_feature_importance(self, sample_df_small, quick_config, quick_selection_config):
        'Test getting feature importance.'
        pipeline = CatBoostPipeline(quick_config, quick_selection_config)
        pipeline.fit(sample_df_small, run_optimization=False, save_artifacts=False)

        importance = pipeline.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns

    def test_save_and_load_artifacts(self, sample_df_small, quick_config, quick_selection_config):
        'Test saving and loading artifacts.'
        pipeline = CatBoostPipeline(quick_config, quick_selection_config)
        pipeline.fit(sample_df_small, run_optimization=False, save_artifacts=True)

        # Load using Scorer
        scorer = Scorer(quick_config.artifacts_dir)
        scorer.load()

        # Predictions should be similar
        pipeline_pred = pipeline.predict(sample_df_small)
        scorer_pred = scorer.score(sample_df_small)

        np.testing.assert_array_almost_equal(pipeline_pred, scorer_pred, decimal=5)

    def test_with_balance_columns(self, sample_df, random_seed, tmp_artifacts_dir):
        'Test with balance columns.'
        config = PipelineConfig(
            id_columns=['client_id', 'app_id'],
            date_column='report_date',
            target_column='target',
            client_column='client_id',
            balance_columns='product_type',
            random_seed=random_seed,
            n_trials=2,
            optuna_timeout=30,
            catboost_iterations=20,
            catboost_early_stopping_rounds=5,
            artifacts_dir=tmp_artifacts_dir
        )

        selection_config = FeatureSelectionConfig(
            run_missing_filter=True,
            run_variance_filter=True,
            run_correlation_filter=False,
            run_psi_filter=False,
            run_importance_filter=False,
        )

        pipeline = CatBoostPipeline(config, selection_config)
        pipeline.fit(sample_df, run_optimization=False, save_artifacts=False)

        metrics = pipeline.get_metrics()

        # Should have group metrics
        assert 'gini_by_group' in metrics['train']

    def test_reproducibility(self, sample_df_small, quick_config, quick_selection_config):
        'Test that results are reproducible with same seed.'
        pipeline1 = CatBoostPipeline(quick_config, quick_selection_config)
        pipeline1.fit(sample_df_small, run_optimization=False, save_artifacts=False)
        pred1 = pipeline1.predict(sample_df_small)

        pipeline2 = CatBoostPipeline(quick_config, quick_selection_config)
        pipeline2.fit(sample_df_small, run_optimization=False, save_artifacts=False)
        pred2 = pipeline2.predict(sample_df_small)

        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_metrics_property(self, sample_df_small, quick_config, quick_selection_config):
        'Test metrics property alias.'
        pipeline = CatBoostPipeline(quick_config, quick_selection_config)
        pipeline.fit(sample_df_small, run_optimization=False, save_artifacts=False)

        metrics_prop = pipeline.metrics
        metrics_method = pipeline.get_metrics()

        assert metrics_prop == metrics_method

    def test_predict_before_fit_raises(self, sample_df_small, quick_config, quick_selection_config):
        'Test that predict before fit raises error.'
        pipeline = CatBoostPipeline(quick_config, quick_selection_config)

        with pytest.raises(RuntimeError):
            pipeline.predict(sample_df_small)

    def test_no_client_overlap_in_splits(self, sample_df, quick_config, quick_selection_config):
        'Test that no client appears in multiple splits.'
        pipeline = CatBoostPipeline(quick_config, quick_selection_config)
        pipeline.fit(sample_df, run_optimization=False, save_artifacts=False)

        split_result = pipeline.split_result_
        client_col = quick_config.client_column

        train_clients = set(split_result.train[client_col])
        valid_clients = set(split_result.valid[client_col])
        oos_clients = set(split_result.oos[client_col])
        oot_clients = set(split_result.oot[client_col])

        assert len(train_clients & valid_clients) == 0
        assert len(train_clients & oos_clients) == 0
        assert len(train_clients & oot_clients) == 0
        assert len(valid_clients & oos_clients) == 0
