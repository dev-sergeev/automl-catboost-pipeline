"Group-aware data splitter for train/valid/OOS/OOT splits."

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import PipelineConfig
from src.utils import get_logger


@dataclass
class SplitResult:
    "Container for split data and indices."

    train: pd.DataFrame
    valid: pd.DataFrame
    oos: pd.DataFrame  # Out-of-sample
    oot: pd.DataFrame  # Out-of-time

    train_idx: np.ndarray
    valid_idx: np.ndarray
    oos_idx: np.ndarray
    oot_idx: np.ndarray

    def get_split(self, name: str) -> pd.DataFrame:
        "Get split by name."
        return getattr(self, name)

    def __repr__(self) -> str:
        return (
            f"SplitResult(train={len(self.train)}, valid={len(self.valid)}, "
            f"oos={len(self.oos)}, oot={len(self.oot)})"
        )


class DataSplitter:
    "Group-aware data splitter ensuring one client is in only one split."

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger("data.splitter")

    def split(self, df: pd.DataFrame) -> SplitResult:
        """
        Split data into train/valid/OOS/OOT with group-aware splitting.

        Algorithm:
        1. Identify OOT: clients with the latest dates (top N% by time)
        2. Split remaining clients into train/valid/OOS using GroupShuffleSplit
        3. All observations of a client go to the same split
        """
        self.logger.info(f"Splitting {len(df):,} rows...")

        # Get configuration
        client_col = self.config.client_column
        date_col = self.config.date_column

        # Validate columns exist
        if client_col not in df.columns:
            raise ValueError(f'Client column "{client_col}" not found in data')
        if date_col not in df.columns:
            raise ValueError(f'Date column "{date_col}" not found in data')

        # Step 1: Identify OOT clients (latest dates)
        oot_idx, non_oot_idx = self._split_oot(df, client_col, date_col)

        # Step 2: Split remaining data into train/valid/OOS
        non_oot_df = df.iloc[non_oot_idx]
        train_idx, valid_idx, oos_idx = self._split_train_valid_oos(
            non_oot_df, client_col, non_oot_idx
        )

        # Create split result
        result = SplitResult(
            train=df.iloc[train_idx].reset_index(drop=True),
            valid=df.iloc[valid_idx].reset_index(drop=True),
            oos=df.iloc[oos_idx].reset_index(drop=True),
            oot=df.iloc[oot_idx].reset_index(drop=True),
            train_idx=train_idx,
            valid_idx=valid_idx,
            oos_idx=oos_idx,
            oot_idx=oot_idx,
        )

        self._log_split_info(result, df)
        self._validate_no_client_overlap(result, client_col)

        return result

    def _split_oot(
        self, df: pd.DataFrame, client_col: str, date_col: str
    ) -> tuple[np.ndarray, np.ndarray]:
        "Split out OOT (Out-of-Time) based on latest dates."
        # Get max date for each client
        client_max_dates = df.groupby(client_col)[date_col].max()

        # Sort clients by their max date
        sorted_clients = client_max_dates.sort_values()

        # Calculate number of clients for OOT
        n_clients = len(sorted_clients)
        n_oot_clients = int(n_clients * self.config.oot_ratio)

        # Get OOT clients (latest dates)
        oot_clients = set(sorted_clients.iloc[-n_oot_clients:].index)

        # Get indices
        oot_mask = df[client_col].isin(oot_clients)
        oot_idx = np.where(oot_mask)[0]
        non_oot_idx = np.where(~oot_mask)[0]

        self.logger.info(
            f"OOT split: {len(oot_clients):,} clients, {len(oot_idx):,} rows"
        )

        return oot_idx, non_oot_idx

    def _split_train_valid_oos(
        self, df: pd.DataFrame, client_col: str, original_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        "Split non-OOT data into train/valid/OOS using GroupShuffleSplit."
        # Calculate ratios relative to non-OOT data
        non_oot_ratio = 1.0 - self.config.oot_ratio
        train_ratio_adj = self.config.train_ratio / non_oot_ratio
        valid_ratio_adj = self.config.valid_ratio / non_oot_ratio
        # oos_ratio_adj = self.config.oos_ratio / non_oot_ratio

        groups = df[client_col].values
        n_samples = len(df)

        # First split: train vs (valid + oos)
        test_size_1 = 1.0 - train_ratio_adj
        gss1 = GroupShuffleSplit(
            n_splits=1, test_size=test_size_1, random_state=self.config.random_seed
        )

        train_local_idx, temp_local_idx = next(
            gss1.split(np.zeros(n_samples), groups=groups)
        )

        # Second split: valid vs oos (from temp)
        temp_groups = groups[temp_local_idx]
        valid_ratio_in_temp = valid_ratio_adj / test_size_1

        gss2 = GroupShuffleSplit(
            n_splits=1,
            test_size=1.0 - valid_ratio_in_temp,
            random_state=self.config.random_seed + 1,
        )

        valid_in_temp_idx, oos_in_temp_idx = next(
            gss2.split(np.zeros(len(temp_local_idx)), groups=temp_groups)
        )

        # Map back to original indices
        valid_local_idx = temp_local_idx[valid_in_temp_idx]
        oos_local_idx = temp_local_idx[oos_in_temp_idx]

        # Map to original DataFrame indices
        train_idx = original_indices[train_local_idx]
        valid_idx = original_indices[valid_local_idx]
        oos_idx = original_indices[oos_local_idx]

        return train_idx, valid_idx, oos_idx

    def _log_split_info(self, result: SplitResult, df: pd.DataFrame) -> None:
        "Log information about the splits."
        total = len(df)
        target_col = self.config.target_column

        for name in ["train", "valid", "oos", "oot"]:
            split_df = result.get_split(name)
            n_rows = len(split_df)
            pct = 100 * n_rows / total

            if target_col in split_df.columns:
                target_rate = split_df[target_col].mean()
                self.logger.info(
                    f"{name.upper()}: {n_rows:,} rows ({pct:.1f}%), "
                    f"target rate: {target_rate:.4f}"
                )
            else:
                self.logger.info(f"{name.upper()}: {n_rows:,} rows ({pct:.1f}%)")

    def _validate_no_client_overlap(self, result: SplitResult, client_col: str) -> None:
        "Validate that no client appears in multiple splits."
        splits = {
            "train": set(result.train[client_col].unique()),
            "valid": set(result.valid[client_col].unique()),
            "oos": set(result.oos[client_col].unique()),
            "oot": set(result.oot[client_col].unique()),
        }

        split_names = list(splits.keys())
        for i, name1 in enumerate(split_names):
            for name2 in split_names[i + 1 :]:
                overlap = splits[name1] & splits[name2]
                if overlap:
                    raise ValueError(
                        f"Client overlap between {name1} and {name2}: "
                        f"{len(overlap)} clients"
                    )

        self.logger.info("Validated: no client overlap between splits")

    def get_split_clients(
        self, result: SplitResult, client_col: Optional[str] = None
    ) -> dict[str, set]:
        "Get unique clients for each split."
        if client_col is None:
            client_col = self.config.client_column

        return {
            "train": set(result.train[client_col].unique()),
            "valid": set(result.valid[client_col].unique()),
            "oos": set(result.oos[client_col].unique()),
            "oot": set(result.oot[client_col].unique()),
        }
