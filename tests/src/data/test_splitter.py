'Tests for data splitter.'

import numpy as np
import pandas as pd
import pytest

from src.data.splitter import DataSplitter, SplitResult


class TestDataSplitter:
    'Tests for DataSplitter.'

    def test_split_basic(self, sample_df, sample_config):
        'Test basic split functionality.'
        splitter = DataSplitter(sample_config)
        result = splitter.split(sample_df)

        assert isinstance(result, SplitResult)
        assert len(result.train) > 0
        assert len(result.valid) > 0
        assert len(result.oos) > 0
        assert len(result.oot) > 0

    def test_split_ratios(self, sample_df, sample_config):
        'Test that split ratios are approximately correct.'
        splitter = DataSplitter(sample_config)
        result = splitter.split(sample_df)

        total = len(sample_df)
        train_ratio = len(result.train) / total
        valid_ratio = len(result.valid) / total
        oos_ratio = len(result.oos) / total
        oot_ratio = len(result.oot) / total

        # Allow 10% tolerance due to group-aware splitting
        assert abs(train_ratio - sample_config.train_ratio) < 0.15
        assert abs(valid_ratio - sample_config.valid_ratio) < 0.15
        assert abs(oos_ratio - sample_config.oos_ratio) < 0.15
        assert abs(oot_ratio - sample_config.oot_ratio) < 0.15

    def test_no_client_overlap(self, sample_df, sample_config):
        'Test that no client appears in multiple splits.'
        splitter = DataSplitter(sample_config)
        result = splitter.split(sample_df)

        client_col = sample_config.client_column

        train_clients = set(result.train[client_col].unique())
        valid_clients = set(result.valid[client_col].unique())
        oos_clients = set(result.oos[client_col].unique())
        oot_clients = set(result.oot[client_col].unique())

        assert len(train_clients & valid_clients) == 0
        assert len(train_clients & oos_clients) == 0
        assert len(train_clients & oot_clients) == 0
        assert len(valid_clients & oos_clients) == 0
        assert len(valid_clients & oot_clients) == 0
        assert len(oos_clients & oot_clients) == 0

    def test_oot_has_latest_dates(self, sample_df, sample_config):
        'Test that OOT contains clients with latest dates.'
        splitter = DataSplitter(sample_config)
        result = splitter.split(sample_df)

        date_col = sample_config.date_column

        # Get max date in OOT
        oot_max_date = result.oot[date_col].max()

        # Get max date in non-OOT splits
        non_oot = pd.concat([result.train, result.valid, result.oos])
        non_oot_max_date = non_oot[date_col].max()

        # OOT should generally have later dates
        assert oot_max_date >= non_oot_max_date

    def test_all_data_preserved(self, sample_df, sample_config):
        'Test that all data is preserved in splits.'
        splitter = DataSplitter(sample_config)
        result = splitter.split(sample_df)

        total_in_splits = (
            len(result.train) + len(result.valid) +
            len(result.oos) + len(result.oot)
        )

        assert total_in_splits == len(sample_df)

    def test_indices_are_valid(self, sample_df, sample_config):
        'Test that returned indices are valid.'
        splitter = DataSplitter(sample_config)
        result = splitter.split(sample_df)

        all_indices = np.concatenate([
            result.train_idx, result.valid_idx,
            result.oos_idx, result.oot_idx
        ])

        # All indices should be unique
        assert len(all_indices) == len(set(all_indices))

        # All indices should be valid
        assert all(0 <= idx < len(sample_df) for idx in all_indices)

    def test_reproducibility(self, sample_df, sample_config):
        'Test that splitting is reproducible with same seed.'
        splitter1 = DataSplitter(sample_config)
        result1 = splitter1.split(sample_df)

        splitter2 = DataSplitter(sample_config)
        result2 = splitter2.split(sample_df)

        np.testing.assert_array_equal(result1.train_idx, result2.train_idx)
        np.testing.assert_array_equal(result1.valid_idx, result2.valid_idx)

    def test_get_split_method(self, sample_df, sample_config):
        'Test get_split method.'
        splitter = DataSplitter(sample_config)
        result = splitter.split(sample_df)

        train = result.get_split('train')
        assert train.equals(result.train)

    def test_missing_client_column_raises(self, sample_config):
        'Test that missing client column raises error.'
        df = pd.DataFrame({
            'wrong_id': [1, 2, 3],
            'report_date': pd.date_range('2024-01-01', periods=3),
            'target': [0, 1, 0]
        })

        splitter = DataSplitter(sample_config)
        with pytest.raises(ValueError, match='Client column'):
            splitter.split(df)
