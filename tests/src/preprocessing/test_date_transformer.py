'Tests for date transformer.'

import numpy as np
import pandas as pd

from src.preprocessing.date_transformer import DateTransformer


class TestDateTransformer:
    'Tests for DateTransformer.'

    def test_init(self):
        'Test transformer initialization.'
        transformer = DateTransformer()
        assert transformer.report_date_column == 'report_date'

        transformer = DateTransformer(report_date_column='custom_date')
        assert transformer.report_date_column == 'custom_date'

    def test_fit_identifies_date_columns(self):
        'Test that fit identifies date columns.'
        df = pd.DataFrame({
            'report_date': pd.date_range('2024-01-01', periods=5),
            'other_date': pd.date_range('2023-01-01', periods=5),
            'num_col': [1, 2, 3, 4, 5],
            'str_col': ['a', 'b', 'c', 'd', 'e']
        })

        transformer = DateTransformer()
        transformer.fit(df)

        # Should identify other_date but not report_date
        assert 'other_date' in transformer.date_columns_
        assert 'report_date' not in transformer.date_columns_
        assert 'num_col' not in transformer.date_columns_
        assert 'str_col' not in transformer.date_columns_

    def test_transform_converts_to_days_diff(self):
        'Test that transform converts dates to days difference.'
        df = pd.DataFrame({
            'report_date': pd.to_datetime(['2024-01-10', '2024-01-15', '2024-01-20']),
            'event_date': pd.to_datetime(['2024-01-05', '2024-01-10', '2024-01-25'])
        })

        transformer = DateTransformer()
        result = transformer.fit_transform(df)

        # event_date should be converted to days from report_date
        expected = np.array([-5, -5, 5])  # days difference
        np.testing.assert_array_equal(result['event_date'].values, expected)

    def test_transform_preserves_report_date(self):
        'Test that report_date column is preserved.'
        df = pd.DataFrame({
            'report_date': pd.date_range('2024-01-01', periods=3),
            'other_date': pd.date_range('2023-12-01', periods=3)
        })

        transformer = DateTransformer()
        result = transformer.fit_transform(df)

        # report_date should be unchanged
        pd.testing.assert_series_equal(
            result['report_date'],
            df['report_date'],
            check_names=False
        )

    def test_string_date_parsing(self):
        'Test parsing string dates.'
        df = pd.DataFrame({
            'report_date': ['2024-01-10', '2024-01-15', '2024-01-20'],
            'event_date': ['2024-01-05', '2024-01-10', '2024-01-25']
        })

        transformer = DateTransformer()
        result = transformer.fit_transform(df)

        expected = np.array([-5, -5, 5])
        np.testing.assert_array_equal(result['event_date'].values, expected)

    def test_missing_report_date_column(self):
        'Test handling when report_date column is missing.'
        df = pd.DataFrame({
            'some_date': pd.date_range('2024-01-01', periods=3)
        })

        transformer = DateTransformer()
        transformer.fit(df)

        # Transform should not fail, just return unchanged
        result = transformer.transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_get_set_params(self):
        'Test serialization of transformer parameters.'
        df = pd.DataFrame({
            'report_date': pd.date_range('2024-01-01', periods=3),
            'event_date': pd.date_range('2023-12-01', periods=3)
        })

        transformer1 = DateTransformer()
        transformer1.fit(df)
        params = transformer1.get_params()

        transformer2 = DateTransformer()
        transformer2.set_params(params)

        assert transformer2.date_columns_ == transformer1.date_columns_
        assert transformer2.report_date_column == transformer1.report_date_column

    def test_handles_nan_dates(self):
        'Test handling of NaN dates.'
        df = pd.DataFrame({
            'report_date': pd.to_datetime(['2024-01-10', '2024-01-15', '2024-01-20']),
            'event_date': pd.to_datetime(['2024-01-05', None, '2024-01-25'])
        })

        transformer = DateTransformer()
        result = transformer.fit_transform(df)

        assert pd.isna(result['event_date'].iloc[1])
        assert result['event_date'].iloc[0] == -5
        assert result['event_date'].iloc[2] == 5
