'Tests for PSITimeFilter.'

import numpy as np
import pandas as pd
import pytest

from src.feature_selection.psi_time_filter import PSITimeFilter


class TestPSITimeFilter:
    'Tests for PSITimeFilter class.'

    @pytest.fixture
    def sample_data(self):
        'Create sample data with time periods.'
        np.random.seed(42)
        n_samples = 1000

        # 60 days of data
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        report_dates = np.random.choice(dates, size=n_samples)

        df = pd.DataFrame({
            'report_date': report_dates,
            'stable_feature': np.random.normal(0, 1, n_samples),
            'category_feature': np.random.choice(['A', 'B', 'C'], n_samples),
        })

        # Add unstable feature that drifts over time
        df = df.sort_values('report_date').reset_index(drop=True)
        drift = np.linspace(0, 3, n_samples)  # Linear drift
        df['unstable_feature'] = np.random.normal(0, 1, n_samples) + drift

        y = pd.Series(np.random.randint(0, 2, n_samples))

        return df, y

    def test_init(self):
        'Test PSITimeFilter initialization.'
        filter_ = PSITimeFilter(
            date_column='report_date',
            threshold=0.25,
            period_days=14
        )

        assert filter_.date_column == 'report_date'
        assert filter_.threshold == 0.25
        assert filter_.period_days == 14
        assert filter_.n_bins == 10
        assert filter_.min_period_days == 14

    def test_fit_detects_unstable_feature(self, sample_data):
        'Test that unstable features are detected.'
        df, y = sample_data

        filter_ = PSITimeFilter(
            date_column='report_date',
            threshold=0.25,
            period_days=14
        )
        filter_.fit(df, y)

        assert filter_.is_fitted_
        assert 'stable_feature' in filter_.selected_features_
        assert 'unstable_feature' in filter_.removed_features_

    def test_fit_keeps_stable_features(self, sample_data):
        'Test that stable features are kept.'
        df, y = sample_data

        filter_ = PSITimeFilter(
            date_column='report_date',
            threshold=0.25,
            period_days=14
        )
        filter_.fit(df, y)

        # Stable feature should be kept
        assert 'stable_feature' in filter_.selected_features_

    def test_transform(self, sample_data):
        'Test transform removes unstable features.'
        df, y = sample_data

        filter_ = PSITimeFilter(
            date_column='report_date',
            threshold=0.25,
            period_days=14
        )
        filter_.fit(df, y)

        df_transformed = filter_.transform(df)

        assert 'stable_feature' in df_transformed.columns
        assert 'unstable_feature' not in df_transformed.columns

    def test_period_splitting(self, sample_data):
        'Test that data is split into correct periods.'
        df, y = sample_data

        filter_ = PSITimeFilter(
            date_column='report_date',
            threshold=0.25,
            period_days=14
        )
        filter_.fit(df, y)

        # 60 days / 14 days = ~4 periods
        assert len(filter_.period_ranges_) >= 3
        assert len(filter_.period_ranges_) <= 5

    def test_last_period_merge(self):
        'Test that short last period is merged.'
        np.random.seed(42)

        # 30 days of data with 14-day periods
        # Should result in 2 periods (last 2 days merged with middle)
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        n_samples = 300

        df = pd.DataFrame({
            'report_date': np.random.choice(dates, size=n_samples),
            'feature': np.random.normal(0, 1, n_samples)
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))

        filter_ = PSITimeFilter(
            date_column='report_date',
            threshold=0.25,
            period_days=14,
            min_period_days=14
        )
        filter_.fit(df, y)

        # 30 days with 14-day periods: period1 (0-14), period2 (14-30)
        # Last period is 16 days which is >= 14, so no merge
        assert len(filter_.period_ranges_) == 2

    def test_get_psi_values(self, sample_data):
        'Test get_psi_values returns correct structure.'
        df, y = sample_data

        filter_ = PSITimeFilter(
            date_column='report_date',
            threshold=0.25,
            period_days=14
        )
        filter_.fit(df, y)

        psi_values = filter_.get_psi_values()

        assert isinstance(psi_values, dict)
        assert 'stable_feature' in psi_values
        assert 'unstable_feature' in psi_values

        # Each feature should have PSI for each comparison period
        for _feature, periods in psi_values.items():
            assert isinstance(periods, dict)
            assert len(periods) > 0

    def test_get_max_psi_values(self, sample_data):
        'Test get_max_psi_values returns maximum PSI.'
        df, y = sample_data

        filter_ = PSITimeFilter(
            date_column='report_date',
            threshold=0.25,
            period_days=14
        )
        filter_.fit(df, y)

        max_psi = filter_.get_max_psi_values()

        # Unstable feature should have higher max PSI
        assert max_psi['unstable_feature'] > max_psi['stable_feature']

    def test_get_psi_dataframe(self, sample_data):
        'Test get_psi_dataframe returns DataFrame.'
        df, y = sample_data

        filter_ = PSITimeFilter(
            date_column='report_date',
            threshold=0.25,
            period_days=14
        )
        filter_.fit(df, y)

        psi_df = filter_.get_psi_dataframe()

        assert isinstance(psi_df, pd.DataFrame)
        assert 'feature' in psi_df.columns
        assert 'max_psi' in psi_df.columns
        # Should be sorted by max_psi descending
        assert psi_df['max_psi'].iloc[0] >= psi_df['max_psi'].iloc[-1]

    def test_with_valid_data(self, sample_data):
        'Test fit with separate validation data.'
        df, y = sample_data

        # Split into train/valid
        train_df = df.iloc[:700]
        valid_df = df.iloc[700:]
        train_y = y.iloc[:700]
        valid_y = y.iloc[700:]

        filter_ = PSITimeFilter(
            date_column='report_date',
            threshold=0.25,
            period_days=14
        )
        filter_.fit(train_df, train_y, X_valid=valid_df, y_valid=valid_y)

        assert filter_.is_fitted_
        assert len(filter_.selected_features_) > 0

    def test_missing_date_column(self, sample_data):
        'Test error when date column is missing.'
        df, y = sample_data
        df_no_date = df.drop(columns=['report_date'])

        filter_ = PSITimeFilter(
            date_column='report_date',
            threshold=0.25,
            period_days=14
        )

        with pytest.raises(ValueError, match='Date column.*not found'):
            filter_.fit(df_no_date, y)

    def test_get_set_params(self, sample_data):
        'Test get_params and set_params.'
        df, y = sample_data

        filter_ = PSITimeFilter(
            date_column='report_date',
            threshold=0.25,
            period_days=14
        )
        filter_.fit(df, y)

        params = filter_.get_params()

        # Create new filter and set params
        new_filter = PSITimeFilter(
            date_column='other_date',
            threshold=0.5,
            period_days=7
        )
        new_filter.set_params(params)

        assert new_filter.threshold == 0.25
        assert new_filter.period_days == 14
        assert new_filter.selected_features_ == filter_.selected_features_
        assert new_filter.removed_features_ == filter_.removed_features_

    def test_categorical_feature_stability(self):
        'Test PSI calculation for categorical features.'
        np.random.seed(42)

        # Create data where category distribution changes over time
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        n_samples = 1000

        df = pd.DataFrame({
            'report_date': np.random.choice(dates, size=n_samples),
        })
        df = df.sort_values('report_date').reset_index(drop=True)

        # Stable categorical
        df['stable_cat'] = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2])

        # Unstable categorical - distribution shifts
        probs_early = [0.5, 0.3, 0.2]
        probs_late = [0.2, 0.3, 0.5]
        n_half = n_samples // 2

        early_cats = np.random.choice(['A', 'B', 'C'], n_half, p=probs_early)
        late_cats = np.random.choice(['A', 'B', 'C'], n_samples - n_half, p=probs_late)
        df['unstable_cat'] = np.concatenate([early_cats, late_cats])

        y = pd.Series(np.random.randint(0, 2, n_samples))

        filter_ = PSITimeFilter(
            date_column='report_date',
            threshold=0.1,  # Lower threshold to catch the shift
            period_days=14
        )
        filter_.fit(df, y)

        # Unstable categorical should have higher PSI
        max_psi = filter_.get_max_psi_values()
        assert max_psi['unstable_cat'] > max_psi['stable_cat']
