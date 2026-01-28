"Data loader for Spark parquet files."

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import pyarrow.parquet as pq

from src.utils import get_logger


class DataLoader:
    "Load data from Spark parquet files."

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = get_logger("data.loader")

    def load_parquet(
        self,
        path: Union[str, Path],
        columns: Optional[list[str]] = None,
        filters: Optional[list] = None,
    ) -> pd.DataFrame:
        "Load parquet file(s) into pandas DataFrame."
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if self.verbose:
            self.logger.info(f"Loading parquet from: {path}")

        # Handle both single file and directory (partitioned parquet)
        if path.is_dir():
            # Spark-style partitioned parquet
            df = self._load_partitioned_parquet(path, columns, filters)
        else:
            # Single parquet file
            df = self._load_single_parquet(path, columns, filters)

        if self.verbose:
            self.logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            self.logger.info(f"Memory usage: {memory_mb:.2f} MB")

        return df

    def _load_single_parquet(
        self,
        path: Path,
        columns: Optional[list[str]] = None,
        filters: Optional[list] = None,
    ) -> pd.DataFrame:
        "Load a single parquet file."
        return pd.read_parquet(path, columns=columns, filters=filters)

    def _load_partitioned_parquet(
        self,
        path: Path,
        columns: Optional[list[str]] = None,
        filters: Optional[list] = None,
    ) -> pd.DataFrame:
        "Load partitioned parquet directory (Spark style)."
        # Use pyarrow for better handling of partitioned datasets
        dataset = pq.ParquetDataset(path, filters=filters)
        table = dataset.read(columns=columns)
        return table.to_pandas()

    def load_multiple(
        self,
        paths: list[Union[str, Path]],
        columns: Optional[list[str]] = None,
        filters: Optional[list] = None,
    ) -> pd.DataFrame:
        "Load and concatenate multiple parquet files."
        dfs = []
        for path in paths:
            df = self.load_parquet(path, columns, filters)
            dfs.append(df)

        result = pd.concat(dfs, ignore_index=True)

        if self.verbose:
            self.logger.info(f"Combined {len(paths)} files: {len(result):,} total rows")

        return result

    def get_schema(self, path: Union[str, Path]) -> dict:
        "Get schema information from parquet file."
        path = Path(path)

        if path.is_dir():
            # Find first parquet file in directory
            parquet_files = list(path.glob("**/*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in: {path}")
            path = parquet_files[0]

        schema = pq.read_schema(path)

        return {
            "columns": [field.name for field in schema],
            "types": {field.name: str(field.type) for field in schema},
            "num_columns": len(schema),
        }

    def get_row_count(self, path: Union[str, Path]) -> int:
        "Get total row count without loading full data."
        path = Path(path)

        if path.is_dir():
            dataset = pq.ParquetDataset(path)
            return sum(pq.read_metadata(f).num_rows for f in dataset.files)
        else:
            return pq.read_metadata(path).num_rows
