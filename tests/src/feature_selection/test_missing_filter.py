'Tests for missing value filter.'

import numpy as np
import pandas as pd

from src.feature_selection.missing_filter import MissingFilter


class TestMissingFilter:
    'Tests for MissingFilter.'

    def test_init(self):
        'Test filter initialization.'
        filter_ = MissingFilter()
        assert filter_.threshold == 0.95

        filter_ = MissingFilter(threshold=0.8)
        assert filter_.threshold == 0.8

    def test_removes_high_missing_features(self):
        'Test that features with high missing rate are removed.'
        df = pd.DataFrame({
            'good_feature': [1, 2, 3, 4, 5],
            'bad_feature': [1, np.nan, np.nan, np.nan, np.nan]  # 80% missing
        })
        y = pd.Series([0, 1, 0, 1, 0])

        filter_ = MissingFilter(threshold=0.5)
        filter_.fit(df, y)

        assert 'good_feature' in filter_.selected_features_
        assert 'bad_feature' in filter_.removed_features_

    def test_keeps_features_below_threshold(self):
        'Test that features below threshold are kept.'
        df = pd.DataFrame({
            'feature_10pct_missing': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],  # 10% missing
            'feature_50pct_missing': [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, 8, 9, 10]  # 50% missing
        })
        y = pd.Series([0, 1] * 5)

        filter_ = MissingFilter(threshold=0.6)
        filter_.fit(df, y)

        assert 'feature_10pct_missing' in filter_.selected_features_
        assert 'feature_50pct_missing' in filter_.selected_features_

    def test_transform(self):
        'Test transform returns selected features.'
        df = pd.DataFrame({
            'good': [1, 2, 3, 4, 5],
            'bad': [np.nan] * 5
        })
        y = pd.Series([0, 1, 0, 1, 0])

        filter_ = MissingFilter(threshold=0.5)
        filter_.fit(df, y)

        result = filter_.transform(df)

        assert list(result.columns) == ['good']

    def test_get_missing_rates(self):
        'Test getting missing rates.'
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, np.nan],  # 20% missing
            'col2': [1, np.nan, np.nan, np.nan, np.nan]  # 80% missing
        })
        y = pd.Series([0, 1, 0, 1, 0])

        filter_ = MissingFilter()
        filter_.fit(df, y)

        rates = filter_.get_missing_rates()

        assert rates['col1'] == 0.2
        assert rates['col2'] == 0.8

    def test_get_set_params(self):
        'Test serialization.'
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [np.nan] * 5
        })
        y = pd.Series([0, 1, 0, 1, 0])

        filter1 = MissingFilter(threshold=0.5)
        filter1.fit(df, y)
        params = filter1.get_params()

        filter2 = MissingFilter()
        filter2.set_params(params)

        assert filter2.selected_features_ == filter1.selected_features_
        assert filter2.removed_features_ == filter1.removed_features_
        assert filter2.threshold == filter1.threshold

    def test_no_missing_values(self):
        'Test with no missing values.'
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [5, 4, 3, 2, 1]
        })
        y = pd.Series([0, 1, 0, 1, 0])

        filter_ = MissingFilter()
        filter_.fit(df, y)

        assert len(filter_.removed_features_) == 0
        assert set(filter_.selected_features_) == {'col1', 'col2'}
