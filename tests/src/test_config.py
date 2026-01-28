'Tests for configuration module.'

import pytest

from src.config import FeatureSelectionConfig, OptunaConfig, PipelineConfig


class TestPipelineConfig:
    'Tests for PipelineConfig.'

    def test_default_config(self):
        'Test default configuration values.'
        config = PipelineConfig()

        assert config.id_columns == ['client_id']
        assert config.date_column == 'report_date'
        assert config.target_column == 'target'
        assert config.random_seed == 42

    def test_id_columns_normalization(self):
        'Test that id_columns is normalized to list.'
        config = PipelineConfig(id_columns='single_id')
        assert config.id_columns == ['single_id']

        config = PipelineConfig(id_columns=['id1', 'id2'])
        assert config.id_columns == ['id1', 'id2']

    def test_client_column_default(self):
        'Test that client_column defaults to first id_column.'
        config = PipelineConfig(id_columns=['client_id', 'app_id'])
        assert config.client_column == 'client_id'

    def test_balance_columns_normalization(self):
        'Test that balance_columns is normalized to list.'
        config = PipelineConfig(balance_columns='product')
        assert config.balance_columns == ['product']

        config = PipelineConfig(balance_columns=['product', 'segment'])
        assert config.balance_columns == ['product', 'segment']

    def test_invalid_ratios(self):
        'Test that invalid split ratios raise error.'
        with pytest.raises(ValueError, match='ratios must sum to 1.0'):
            PipelineConfig(
                train_ratio=0.5,
                valid_ratio=0.2,
                oos_ratio=0.1,
                oot_ratio=0.1  # Sum = 0.9
            )

    def test_valid_ratios(self):
        'Test valid split ratios.'
        config = PipelineConfig(
            train_ratio=0.7,
            valid_ratio=0.15,
            oos_ratio=0.1,
            oot_ratio=0.05
        )
        total = config.train_ratio + config.valid_ratio + config.oos_ratio + config.oot_ratio
        assert abs(total - 1.0) < 1e-6

    def test_invalid_thresholds(self):
        'Test that invalid thresholds raise error.'
        with pytest.raises(ValueError, match='missing_threshold'):
            PipelineConfig(missing_threshold=1.5)

        with pytest.raises(ValueError, match='correlation_threshold'):
            PipelineConfig(correlation_threshold=-0.1)

        with pytest.raises(ValueError, match='psi_threshold'):
            PipelineConfig(psi_threshold=-0.5)

    def test_get_id_columns_list(self):
        'Test get_id_columns_list method.'
        config = PipelineConfig(id_columns='single_id')
        assert config.get_id_columns_list() == ['single_id']

    def test_get_balance_columns_list(self):
        'Test get_balance_columns_list method.'
        config = PipelineConfig(balance_columns=None)
        assert config.get_balance_columns_list() is None

        config = PipelineConfig(balance_columns='product')
        assert config.get_balance_columns_list() == ['product']


class TestFeatureSelectionConfig:
    'Tests for FeatureSelectionConfig.'

    def test_default_config(self):
        'Test default values.'
        config = FeatureSelectionConfig()

        assert config.run_missing_filter is True
        assert config.run_variance_filter is True
        assert config.run_backward_selection is False
        assert config.run_forward_selection is False

    def test_custom_thresholds(self):
        'Test custom threshold settings.'
        config = FeatureSelectionConfig(
            missing_threshold=0.8,
            psi_threshold=0.3
        )

        assert config.missing_threshold == 0.8
        assert config.psi_threshold == 0.3


class TestOptunaConfig:
    'Tests for OptunaConfig.'

    def test_default_config(self):
        'Test default values.'
        config = OptunaConfig()

        assert config.n_trials == 100
        assert config.timeout == 3600
        assert config.use_pruning is True

    def test_search_space_bounds(self):
        'Test search space bounds.'
        config = OptunaConfig(
            iterations_min=50,
            iterations_max=500
        )

        assert config.iterations_min == 50
        assert config.iterations_max == 500
