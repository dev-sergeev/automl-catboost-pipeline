'Tests for main preprocessor.'

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.preprocessing import Preprocessor


class TestPreprocessor:
    'Tests for Preprocessor.'

    def test_init(self, sample_config):
        'Test preprocessor initialization.'
        preprocessor = Preprocessor(sample_config)

        assert preprocessor.decimal_handler is not None
        assert preprocessor.date_transformer is not None
        assert preprocessor.type_detector is not None

    def test_fit_transform(self, sample_df, sample_config):
        'Test fit and transform.'
        preprocessor = Preprocessor(sample_config)

        result = preprocessor.fit_transform(sample_df)

        assert preprocessor.is_fitted_
        assert len(result) == len(sample_df)

    def test_feature_columns_computed(self, sample_df, sample_config):
        'Test that feature columns are correctly computed.'
        preprocessor = Preprocessor(sample_config)
        preprocessor.fit(sample_df)

        features = preprocessor.get_feature_columns()

        # Should not include id, date, target columns
        assert 'client_id' not in features
        assert 'app_id' not in features
        assert 'report_date' not in features
        assert 'target' not in features

        # Should include feature columns
        assert 'num_feature_1' in features

    def test_categorical_features(self, sample_df, sample_config):
        'Test categorical feature detection.'
        preprocessor = Preprocessor(sample_config)
        preprocessor.fit(sample_df)

        cat_features = preprocessor.get_categorical_features()

        assert 'cat_feature_1' in cat_features
        assert 'cat_feature_2' in cat_features

    def test_numeric_features(self, sample_df, sample_config):
        'Test numeric feature detection.'
        preprocessor = Preprocessor(sample_config)
        preprocessor.fit(sample_df)

        num_features = preprocessor.get_numeric_features()

        assert 'num_feature_1' in num_features
        assert 'num_feature_2' in num_features

    def test_decimal_handling(self, sample_config):
        'Test handling of Decimal types.'
        df = pd.DataFrame({
            'client_id': [1, 2, 3],
            'report_date': pd.date_range('2024-01-01', periods=3),
            'target': [0, 1, 0],
            'decimal_col': [Decimal('1.5'), Decimal('2.5'), Decimal('3.5')]
        })

        preprocessor = Preprocessor(sample_config)
        result = preprocessor.fit_transform(df)

        assert result['decimal_col'].dtype == np.float64

    def test_date_transformation(self, sample_config):
        'Test date column transformation.'
        df = pd.DataFrame({
            'client_id': [1, 2, 3],
            'report_date': pd.to_datetime(['2024-01-10', '2024-01-15', '2024-01-20']),
            'target': [0, 1, 0],
            'event_date': pd.to_datetime(['2024-01-05', '2024-01-10', '2024-01-25'])
        })

        preprocessor = Preprocessor(sample_config)
        result = preprocessor.fit_transform(df)

        # event_date should be converted to days difference
        assert result['event_date'].dtype == np.float64
        assert result['event_date'].iloc[0] == -5  # 5 days before report_date

    def test_get_set_params(self, sample_df, sample_config):
        'Test serialization of preprocessor parameters.'
        preprocessor1 = Preprocessor(sample_config)
        preprocessor1.fit(sample_df)
        params = preprocessor1.get_params()

        preprocessor2 = Preprocessor(sample_config)
        preprocessor2.set_params(params)

        assert preprocessor2.feature_columns_ == preprocessor1.feature_columns_
        assert preprocessor2.is_fitted_

    def test_transform_before_fit_raises(self, sample_df, sample_config):
        'Test that transform before fit raises error.'
        preprocessor = Preprocessor(sample_config)

        with pytest.raises(RuntimeError, match='fitted'):
            preprocessor.transform(sample_df)

    def test_balance_columns_excluded(self, random_seed):
        'Test that balance columns are excluded from features.'
        df = pd.DataFrame({
            'client_id': [1, 2, 3],
            'report_date': pd.date_range('2024-01-01', periods=3),
            'target': [0, 1, 0],
            'product_type': ['A', 'B', 'A'],
            'feature1': [1.0, 2.0, 3.0]
        })

        config = PipelineConfig(
            id_columns='client_id',
            balance_columns='product_type',
            random_seed=random_seed
        )

        preprocessor = Preprocessor(config)
        preprocessor.fit(df)

        features = preprocessor.get_feature_columns()

        assert 'product_type' not in features
        assert 'feature1' in features
