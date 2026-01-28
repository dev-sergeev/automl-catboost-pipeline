'Utility functions for the pipeline.'

import logging
import random
from typing import Optional

import numpy as np


class NoFeaturesRemainingError(Exception):
    '''
    Raised when feature selection removes all features.

    Contains details about which filters removed features and recommendations
    for resolving the issue.
    '''

    def __init__(
        self,
        message: str,
        selection_history: list[dict],
        last_filter: str
    ):
        self.selection_history = selection_history
        self.last_filter = last_filter
        super().__init__(message)

    def get_detailed_report(self) -> str:
        'Generate a detailed report of feature removal.'
        lines = [
            '',
            '=' * 70,
            'FEATURE SELECTION FAILED: No features passed quality thresholds',
            '=' * 70,
            '',
            'Selection history:',
        ]

        for step in self.selection_history:
            step_name = step['step']
            removed = step['removed']
            remaining = step['remaining']
            lines.append(f'  {step_name}: removed {removed}, remaining {remaining}')

        lines.extend([
            '',
            f'All features were removed at step: {self.last_filter}',
            '',
            'Recommendations:',
            '  1. Review your data quality - features may have too many missing values',
            '  2. Adjust thresholds in FeatureSelectionConfig:',
            '     - missing_threshold: increase if too many features removed by MissingFilter',
            '     - variance_threshold: decrease if VarianceFilter removes too many',
            '     - correlation_threshold: decrease if CorrelationFilter removes too many',
            '     - psi_threshold: increase if PSI filters remove too many',
            '  3. Disable specific filters that are too aggressive:',
            '     - run_missing_filter=False',
            '     - run_variance_filter=False',
            '     - run_correlation_filter=False',
            '     - run_psi_filter=False',
            '     - run_psi_time_filter=False',
            '     - run_importance_filter=False',
            '  4. Check the selection_history_ attribute for detailed removal info',
            '',
            '=' * 70,
        ])
        return '\n'.join(lines)


def setup_logging(
    level: int = logging.INFO,
    format_str: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    'Set up logging configuration and return root logger.'
    logging.basicConfig(level=level, format=format_str)
    return logging.getLogger('automl')


def get_logger(name: str) -> logging.Logger:
    'Get a logger with the given name.'
    return logging.getLogger(f'automl.{name}')


def set_random_seed(seed: int) -> None:
    'Set random seed for reproducibility across all libraries.'
    random.seed(seed)
    np.random.seed(seed)

    # Try to set seed for other libraries if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def calculate_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    'Calculate Gini coefficient from AUC.'
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_pred)
    return 2 * auc - 1


def calculate_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    'Calculate AUC-ROC score.'
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred)


def ensure_list(value: Optional[object]) -> list:
    'Ensure value is a list.'
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def memory_usage_mb(df) -> float:
    'Return memory usage of DataFrame in MB.'
    return df.memory_usage(deep=True).sum() / 1024 / 1024


def reduce_memory_usage(df, verbose: bool = True):
    'Reduce memory usage of DataFrame by downcasting numeric types.'
    logger = get_logger('utils')
    start_mem = memory_usage_mb(df)

    for col in df.columns:
        col_type = df[col].dtype

        if col_type is not object and str(col_type) != 'category':
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = memory_usage_mb(df)

    if verbose:
        logger.info(
            f'Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB '
            f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)'
        )

    return df


def get_feature_columns(
    df,
    id_columns: list[str],
    date_column: str,
    target_column: str,
    balance_columns: Optional[list[str]] = None
) -> list[str]:
    'Get list of feature columns (excluding id, date, target, balance columns).'
    exclude_cols = set(id_columns)
    exclude_cols.add(date_column)
    exclude_cols.add(target_column)
    if balance_columns:
        exclude_cols.update(balance_columns)

    return [col for col in df.columns if col not in exclude_cols]


def validate_dataframe(
    df,
    required_columns: list[str],
    name: str = 'DataFrame'
) -> None:
    'Validate that DataFrame contains required columns.'
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')
