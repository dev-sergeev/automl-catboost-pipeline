'Sample balancer for computing sample weights by groups.'

from typing import Optional

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.utils import get_logger


class SampleBalancer:
    '''
    Compute sample weights for balancing by groups.

    When data has strong imbalance by groups (e.g., products: ПК 80%, КК 18%, ИК 2%),
    balancing helps prevent loss of quality on rare groups.

    Weights are computed inversely proportional to group frequency and normalized
    so that mean weight = 1.0.
    '''

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger('data.balancer')
        self.group_weights_: Optional[dict] = None
        self.is_fitted_: bool = False

    @property
    def balance_columns(self) -> Optional[list[str]]:
        'Get balance columns as list.'
        return self.config.get_balance_columns_list()

    def fit(self, df: pd.DataFrame) -> 'SampleBalancer':
        'Fit balancer by computing group frequencies.'
        if self.balance_columns is None:
            self.logger.info('No balance columns specified, skipping balancing')
            self.is_fitted_ = True
            return self

        self.logger.info(f'Fitting balancer on columns: {self.balance_columns}')

        # Create composite group key if multiple columns
        group_key = self._get_group_key(df)

        # Compute group frequencies
        group_counts = group_key.value_counts(normalize=True)

        # Compute weights (inverse of frequency)
        group_weights = 1.0 / group_counts

        # Normalize to mean = 1.0
        group_weights = group_weights / group_weights.mean()

        self.group_weights_ = group_weights.to_dict()
        self.is_fitted_ = True

        # Log weight distribution
        self._log_weights()

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        'Transform DataFrame to sample weights array.'
        if not self.is_fitted_:
            raise RuntimeError('Balancer must be fitted before transform')

        if self.balance_columns is None:
            # Return uniform weights
            return np.ones(len(df))

        group_key = self._get_group_key(df)

        # Map group keys to weights
        weights = group_key.map(self.group_weights_)

        # Handle unknown groups (use weight = 1.0)
        unknown_mask = weights.isna()
        if unknown_mask.any():
            n_unknown = unknown_mask.sum()
            self.logger.warning(
                f'Found {n_unknown} rows with unknown groups, using weight=1.0'
            )
            weights = weights.fillna(1.0)

        return weights.values

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        'Fit and transform in one step.'
        return self.fit(df).transform(df)

    def _get_group_key(self, df: pd.DataFrame) -> pd.Series:
        'Create composite group key from balance columns.'
        if len(self.balance_columns) == 1:
            return df[self.balance_columns[0]].astype(str)
        else:
            # Combine multiple columns into single key
            return df[self.balance_columns].astype(str).agg('|'.join, axis=1)

    def _log_weights(self) -> None:
        'Log weight distribution information.'
        if self.group_weights_ is None:
            return

        weights_series = pd.Series(self.group_weights_)
        self.logger.info(f'Computed weights for {len(weights_series)} groups:')

        # Sort by weight descending (rare groups have higher weights)
        sorted_weights = weights_series.sort_values(ascending=False)

        # Log top groups (highest weights = rarest)
        for group, weight in sorted_weights.head(10).items():
            freq = 1.0 / weight if weight > 0 else 0
            self.logger.info(f'  {group}: weight={weight:.3f} (freq={freq:.4f})')

        if len(sorted_weights) > 10:
            self.logger.info(f'  ... and {len(sorted_weights) - 10} more groups')

    def get_group_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        'Get statistics by group.'
        if self.balance_columns is None:
            return pd.DataFrame()

        group_key = self._get_group_key(df)
        target_col = self.config.target_column

        stats = df.groupby(group_key).agg(
            count=(target_col, 'count'),
            target_rate=(target_col, 'mean')
        ).reset_index()

        stats.columns = ['group', 'count', 'target_rate']
        stats['frequency'] = stats['count'] / stats['count'].sum()

        if self.group_weights_:
            stats['weight'] = stats['group'].map(self.group_weights_)

        return stats.sort_values('count', ascending=False)

    def get_params(self) -> dict:
        'Get balancer parameters for serialization.'
        return {
            'balance_columns': self.balance_columns,
            'group_weights': self.group_weights_,
            'is_fitted': self.is_fitted_
        }

    def set_params(self, params: dict) -> 'SampleBalancer':
        'Set balancer parameters from serialization.'
        self.group_weights_ = params.get('group_weights')
        self.is_fitted_ = params.get('is_fitted', False)
        return self
