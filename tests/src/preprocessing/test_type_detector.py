'Tests for type detector.'

import numpy as np
import pandas as pd

from src.preprocessing.type_detector import TypeDetector


class TestTypeDetector:
    'Tests for TypeDetector.'

    def test_init(self):
        'Test detector initialization.'
        detector = TypeDetector()
        assert detector.unique_threshold == 20

        detector = TypeDetector(unique_threshold=50)
        assert detector.unique_threshold == 50

    def test_numeric_columns_detected(self):
        'Test that numeric columns are detected correctly.'
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        detector = TypeDetector()
        detector.fit(df)

        assert 'int_col' in detector.numeric_columns_
        assert 'float_col' in detector.numeric_columns_

    def test_low_cardinality_string_is_categorical(self):
        'Test that low cardinality strings are categorical.'
        df = pd.DataFrame({
            'cat_col': ['A', 'B', 'C', 'A', 'B']
        })

        detector = TypeDetector(unique_threshold=10)
        detector.fit(df)

        assert 'cat_col' in detector.categorical_columns_
        assert 'cat_col' not in detector.numeric_columns_

    def test_high_cardinality_numeric_string(self):
        'Test that high cardinality numeric strings become numeric.'
        df = pd.DataFrame({
            'num_str_col': [str(i) for i in range(100)]
        })

        detector = TypeDetector(unique_threshold=20)
        detector.fit(df)

        # High cardinality numeric strings should be numeric
        assert 'num_str_col' in detector.numeric_columns_

    def test_high_cardinality_non_numeric_string(self):
        'Test that high cardinality non-numeric strings stay categorical.'
        df = pd.DataFrame({
            'str_col': [f'value_{i}' for i in range(100)]
        })

        detector = TypeDetector(unique_threshold=20)
        detector.fit(df)

        # High cardinality non-numeric strings -> categorical
        assert 'str_col' in detector.categorical_columns_

    def test_exclude_columns(self):
        'Test that excluded columns are not processed.'
        df = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5],
            'id': [100, 200, 300, 400, 500]
        })

        detector = TypeDetector(exclude_columns=['id'])
        detector.fit(df)

        assert 'feature' in detector.numeric_columns_
        assert 'id' not in detector.numeric_columns_
        assert 'id' not in detector.categorical_columns_

    def test_transform_numeric(self):
        'Test transformation of numeric columns.'
        df = pd.DataFrame({
            'num_col': ['1', '2', '3', '4', '5']
        })

        detector = TypeDetector(unique_threshold=3)
        result = detector.fit_transform(df)

        assert result['num_col'].dtype == np.float64

    def test_transform_categorical(self):
        'Test transformation of categorical columns.'
        df = pd.DataFrame({
            'cat_col': ['A', 'B', None, 'A', 'B']
        })

        detector = TypeDetector()
        result = detector.fit_transform(df)

        # NaN should be replaced with string
        # In pandas 3.x, string columns may be str dtype instead of object
        assert result['cat_col'].dtype == object or pd.api.types.is_string_dtype(result['cat_col'].dtype)
        assert result['cat_col'].isna().sum() == 0

    def test_get_feature_types(self):
        'Test get_feature_types method.'
        df = pd.DataFrame({
            'num_col': [1.0, 2.0, 3.0],
            'cat_col': ['A', 'B', 'A']
        })

        detector = TypeDetector()
        detector.fit(df)

        types = detector.get_feature_types()

        assert 'numeric' in types
        assert 'categorical' in types
        assert 'num_col' in types['numeric']
        assert 'cat_col' in types['categorical']

    def test_get_catboost_cat_features(self):
        'Test getting categorical features for CatBoost.'
        df = pd.DataFrame({
            'num': [1, 2, 3],
            'cat1': ['A', 'B', 'C'],
            'cat2': ['X', 'Y', 'Z']
        })

        detector = TypeDetector()
        detector.fit(df)

        cat_features = detector.get_catboost_cat_features()

        assert 'cat1' in cat_features
        assert 'cat2' in cat_features
        assert 'num' not in cat_features

    def test_get_set_params(self):
        'Test serialization of detector parameters.'
        df = pd.DataFrame({
            'num_col': [1.0, 2.0, 3.0],
            'cat_col': ['A', 'B', 'A']
        })

        detector1 = TypeDetector()
        detector1.fit(df)
        params = detector1.get_params()

        detector2 = TypeDetector()
        detector2.set_params(params)

        assert detector2.numeric_columns_ == detector1.numeric_columns_
        assert detector2.categorical_columns_ == detector1.categorical_columns_
        assert detector2.column_types_ == detector1.column_types_

    def test_boolean_columns_are_categorical(self):
        'Test that boolean columns become categorical.'
        # Explicitly create bool dtype
        df = pd.DataFrame({
            'bool_col': pd.array([True, False, True, False, True], dtype='bool')
        })

        detector = TypeDetector()
        detector.fit(df)

        assert 'bool_col' in detector.categorical_columns_
