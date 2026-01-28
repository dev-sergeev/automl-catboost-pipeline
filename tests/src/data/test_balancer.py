'Tests for sample balancer.'

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.data.balancer import SampleBalancer


class TestSampleBalancer:
    'Tests for SampleBalancer.'

    def test_no_balance_columns(self, sample_df_small, random_seed):
        'Test balancer when no balance columns specified.'
        config = PipelineConfig(
            id_columns='client_id',
            balance_columns=None,
            random_seed=random_seed
        )

        balancer = SampleBalancer(config)
        weights = balancer.fit_transform(sample_df_small)

        # All weights should be 1.0 when no balancing
        assert np.allclose(weights, 1.0)

    def test_single_balance_column(self, random_seed):
        'Test balancing with single column.'
        # Create imbalanced data
        df = pd.DataFrame({
            'client_id': range(100),
            'product': ['A'] * 80 + ['B'] * 15 + ['C'] * 5,
            'target': np.random.randint(0, 2, 100)
        })

        config = PipelineConfig(
            id_columns='client_id',
            balance_columns='product',
            random_seed=random_seed
        )

        balancer = SampleBalancer(config)
        weights = balancer.fit_transform(df)

        # Rare groups should have higher weights
        weight_c = weights[df['product'] == 'C'].mean()
        weight_a = weights[df['product'] == 'A'].mean()

        assert weight_c > weight_a

    def test_weights_mean_normalized(self, random_seed):
        'Test that weights are normalized to mean=1.'
        df = pd.DataFrame({
            'client_id': range(100),
            'product': ['A'] * 50 + ['B'] * 50,
            'target': np.random.randint(0, 2, 100)
        })

        config = PipelineConfig(
            id_columns='client_id',
            balance_columns='product',
            random_seed=random_seed
        )

        balancer = SampleBalancer(config)
        weights = balancer.fit_transform(df)

        assert np.isclose(weights.mean(), 1.0, atol=0.01)

    def test_multiple_balance_columns(self, random_seed):
        'Test balancing with multiple columns.'
        df = pd.DataFrame({
            'client_id': range(100),
            'product': ['A'] * 70 + ['B'] * 30,
            'segment': ['X'] * 50 + ['Y'] * 50,
            'target': np.random.randint(0, 2, 100)
        })

        config = PipelineConfig(
            id_columns='client_id',
            balance_columns=['product', 'segment'],
            random_seed=random_seed
        )

        balancer = SampleBalancer(config)
        weights = balancer.fit_transform(df)

        assert len(weights) == len(df)
        # Weights are positive
        assert np.all(weights > 0)

    def test_transform_new_data(self, random_seed):
        'Test transforming new data after fit.'
        train_df = pd.DataFrame({
            'client_id': range(100),
            'product': ['A'] * 80 + ['B'] * 20,
            'target': np.random.randint(0, 2, 100)
        })

        test_df = pd.DataFrame({
            'client_id': range(50),
            'product': ['A'] * 25 + ['B'] * 25,
            'target': np.random.randint(0, 2, 50)
        })

        config = PipelineConfig(
            id_columns='client_id',
            balance_columns='product',
            random_seed=random_seed
        )

        balancer = SampleBalancer(config)
        balancer.fit(train_df)

        weights = balancer.transform(test_df)
        assert len(weights) == len(test_df)

    def test_unknown_group_handling(self, random_seed):
        'Test handling of unknown groups in transform.'
        train_df = pd.DataFrame({
            'client_id': range(100),
            'product': ['A'] * 50 + ['B'] * 50,
            'target': np.random.randint(0, 2, 100)
        })

        test_df = pd.DataFrame({
            'client_id': range(10),
            'product': ['C'] * 10,  # New group not in training
            'target': np.random.randint(0, 2, 10)
        })

        config = PipelineConfig(
            id_columns='client_id',
            balance_columns='product',
            random_seed=random_seed
        )

        balancer = SampleBalancer(config)
        balancer.fit(train_df)

        # Should not raise, unknown groups get weight=1.0
        weights = balancer.transform(test_df)
        assert np.allclose(weights, 1.0)

    def test_get_group_stats(self, random_seed):
        'Test get_group_stats method.'
        df = pd.DataFrame({
            'client_id': range(100),
            'product': ['A'] * 60 + ['B'] * 40,
            'target': [1] * 20 + [0] * 40 + [1] * 10 + [0] * 30
        })

        config = PipelineConfig(
            id_columns='client_id',
            balance_columns='product',
            random_seed=random_seed
        )

        balancer = SampleBalancer(config)
        balancer.fit(df)

        stats = balancer.get_group_stats(df)

        assert 'group' in stats.columns
        assert 'count' in stats.columns
        assert 'target_rate' in stats.columns
        assert 'frequency' in stats.columns
        assert 'weight' in stats.columns
        assert len(stats) == 2

    def test_get_set_params(self, random_seed):
        'Test serialization of balancer parameters.'
        df = pd.DataFrame({
            'client_id': range(100),
            'product': ['A'] * 50 + ['B'] * 50,
            'target': np.random.randint(0, 2, 100)
        })

        config = PipelineConfig(
            id_columns='client_id',
            balance_columns='product',
            random_seed=random_seed
        )

        balancer1 = SampleBalancer(config)
        balancer1.fit(df)
        params = balancer1.get_params()

        balancer2 = SampleBalancer(config)
        balancer2.set_params(params)

        weights1 = balancer1.transform(df)
        weights2 = balancer2.transform(df)

        np.testing.assert_array_almost_equal(weights1, weights2)
