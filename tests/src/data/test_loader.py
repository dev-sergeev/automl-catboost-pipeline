'Tests for data loader.'

import pandas as pd
import pytest

from src.data.loader import DataLoader


class TestDataLoader:
    'Tests for DataLoader.'

    def test_init(self):
        'Test loader initialization.'
        loader = DataLoader()
        assert loader.verbose is True

        loader = DataLoader(verbose=False)
        assert loader.verbose is False

    def test_load_parquet_file_not_found(self, tmp_path):
        'Test loading non-existent file raises error.'
        loader = DataLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_parquet(tmp_path / 'nonexistent.parquet')

    def test_load_parquet(self, tmp_path):
        'Test loading a parquet file.'
        # Create test parquet file
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        })
        file_path = tmp_path / 'test.parquet'
        df.to_parquet(file_path)

        loader = DataLoader()
        loaded_df = loader.load_parquet(file_path)

        pd.testing.assert_frame_equal(df, loaded_df)

    def test_load_parquet_with_columns(self, tmp_path):
        'Test loading specific columns.'
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z'],
            'c': [1.0, 2.0, 3.0]
        })
        file_path = tmp_path / 'test.parquet'
        df.to_parquet(file_path)

        loader = DataLoader()
        loaded_df = loader.load_parquet(file_path, columns=['a', 'b'])

        assert list(loaded_df.columns) == ['a', 'b']
        assert 'c' not in loaded_df.columns

    def test_get_schema(self, tmp_path):
        'Test getting schema from parquet file.'
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'str_col': ['x', 'y', 'z'],
            'float_col': [1.0, 2.0, 3.0]
        })
        file_path = tmp_path / 'test.parquet'
        df.to_parquet(file_path)

        loader = DataLoader()
        schema = loader.get_schema(file_path)

        assert 'columns' in schema
        assert 'types' in schema
        assert 'num_columns' in schema
        assert schema['num_columns'] == 3
        assert set(schema['columns']) == {'int_col', 'str_col', 'float_col'}

    def test_get_row_count(self, tmp_path):
        'Test getting row count without loading data.'
        df = pd.DataFrame({
            'a': range(100)
        })
        file_path = tmp_path / 'test.parquet'
        df.to_parquet(file_path)

        loader = DataLoader()
        row_count = loader.get_row_count(file_path)

        assert row_count == 100

    def test_load_multiple(self, tmp_path):
        'Test loading multiple parquet files.'
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'a': [4, 5, 6]})

        file1 = tmp_path / 'test1.parquet'
        file2 = tmp_path / 'test2.parquet'

        df1.to_parquet(file1)
        df2.to_parquet(file2)

        loader = DataLoader()
        combined = loader.load_multiple([file1, file2])

        assert len(combined) == 6
        assert list(combined['a']) == [1, 2, 3, 4, 5, 6]
