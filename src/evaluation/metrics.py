'Evaluation metrics calculation.'

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.utils import get_logger


class MetricsCalculator:
    'Calculate evaluation metrics for binary classification.'

    def __init__(self):
        self.logger = get_logger('evaluation.metrics')

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> dict:
        '''
        Calculate standard metrics.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            sample_weight: Optional sample weights

        Returns:
            Dictionary with metrics
        '''
        metrics = {}

        # AUC-ROC
        try:
            metrics['auc'] = float(roc_auc_score(
                y_true, y_pred, sample_weight=sample_weight
            ))
        except ValueError:
            metrics['auc'] = np.nan

        # Gini coefficient (2 * AUC - 1)
        metrics['gini'] = 2 * metrics['auc'] - 1 if not np.isnan(metrics['auc']) else np.nan

        # Average Precision (PR-AUC)
        try:
            metrics['avg_precision'] = float(average_precision_score(
                y_true, y_pred, sample_weight=sample_weight
            ))
        except ValueError:
            metrics['avg_precision'] = np.nan

        # Count statistics
        metrics['n_samples'] = int(len(y_true))
        metrics['n_positive'] = int(np.sum(y_true))
        metrics['n_negative'] = int(len(y_true) - np.sum(y_true))
        metrics['target_rate'] = float(np.mean(y_true))

        return metrics

    def calculate_metrics_by_group(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> dict[str, dict]:
        '''
        Calculate metrics for each group.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            groups: Group labels (e.g., product type)
            sample_weight: Optional sample weights

        Returns:
            Dictionary with metrics per group
        '''
        results = {}

        for group in groups.unique():
            mask = groups == group
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]

            if sample_weight is not None:
                group_weight = sample_weight[mask]
            else:
                group_weight = None

            if len(np.unique(group_y_true)) < 2:
                # Skip if only one class present
                results[str(group)] = {
                    'auc': np.nan,
                    'gini': np.nan,
                    'n_samples': len(group_y_true),
                    'target_rate': float(np.mean(group_y_true)) if len(group_y_true) > 0 else 0.0
                }
                continue

            results[str(group)] = self.calculate_metrics(
                group_y_true, group_y_pred, group_weight
            )

        return results

    def calculate_metrics_by_time(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: pd.Series,
        freq: str = 'M'
    ) -> pd.DataFrame:
        '''
        Calculate metrics over time periods.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            dates: Date column
            freq: Frequency for grouping ('M' for monthly, 'W' for weekly)

        Returns:
            DataFrame with metrics per time period
        '''
        # Create period column
        dates = pd.to_datetime(dates)
        periods = dates.dt.to_period(freq)

        results = []
        for period in periods.unique():
            mask = periods == period
            period_y_true = y_true[mask]
            period_y_pred = y_pred[mask]

            if len(np.unique(period_y_true)) < 2:
                metrics = {
                    'auc': np.nan,
                    'gini': np.nan,
                    'n_samples': len(period_y_true),
                    'target_rate': float(np.mean(period_y_true)) if len(period_y_true) > 0 else 0.0
                }
            else:
                metrics = self.calculate_metrics(period_y_true, period_y_pred)

            metrics['period'] = str(period)
            results.append(metrics)

        df = pd.DataFrame(results)
        return df.sort_values('period').reset_index(drop=True)

    def calculate_all_splits_metrics(
        self,
        splits_data: dict[str, tuple[np.ndarray, np.ndarray]],
        groups_data: Optional[dict[str, pd.Series]] = None
    ) -> dict:
        '''
        Calculate metrics for all data splits.

        Args:
            splits_data: Dictionary with split name -> (y_true, y_pred)
            groups_data: Optional dictionary with split name -> groups series

        Returns:
            Dictionary with metrics for each split
        '''
        results = {}

        for split_name, (y_true, y_pred) in splits_data.items():
            self.logger.info(f'Calculating metrics for {split_name}...')

            # Overall metrics
            metrics = self.calculate_metrics(y_true, y_pred)

            # Metrics by group if available
            if groups_data and split_name in groups_data:
                groups = groups_data[split_name]
                metrics['gini_by_group'] = {
                    g: m.get('gini', np.nan)
                    for g, m in self.calculate_metrics_by_group(
                        y_true, y_pred, groups
                    ).items()
                }

            results[split_name] = metrics

        return results

    def compare_gini_by_group(
        self,
        metrics_dict: dict
    ) -> pd.DataFrame:
        '''
        Create comparison table of Gini by group across splits.

        Args:
            metrics_dict: Dictionary from calculate_all_splits_metrics

        Returns:
            DataFrame with groups as rows, splits as columns
        '''
        # Collect all groups
        all_groups = set()
        for split_metrics in metrics_dict.values():
            if 'gini_by_group' in split_metrics:
                all_groups.update(split_metrics['gini_by_group'].keys())

        if not all_groups:
            return pd.DataFrame()

        # Build comparison table
        rows = []
        for group in sorted(all_groups):
            row = {'group': group}
            for split_name, split_metrics in metrics_dict.items():
                gini_by_group = split_metrics.get('gini_by_group', {})
                row[f'gini_{split_name}'] = gini_by_group.get(group, np.nan)
            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def gini_from_auc(auc: float) -> float:
        'Convert AUC to Gini coefficient.'
        return 2 * auc - 1

    @staticmethod
    def auc_from_gini(gini: float) -> float:
        'Convert Gini coefficient to AUC.'
        return (gini + 1) / 2
