'Plots for model metrics visualization.'

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import get_logger


class MetricsPlotter:
    'Create plots for model evaluation metrics.'

    def __init__(self, figsize: tuple[int, int] = (12, 6)):
        self.figsize = figsize
        self.logger = get_logger('visualization.metrics')

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette('husl')

    def plot_gini_by_splits(
        self,
        metrics: dict,
        title: str = 'Gini by Data Splits',
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        '''
        Bar chart of Gini coefficient by data split.

        Args:
            metrics: Dictionary with split name -> metrics dict
            title: Plot title
            ax: Optional axes to plot on

        Returns:
            Matplotlib figure
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        splits = list(metrics.keys())
        ginis = [metrics[s].get('gini', 0) for s in splits]

        # Define colors for different splits
        colors = {
            'train': '#2ecc71',
            'valid': '#3498db',
            'oos': '#e74c3c',
            'oot': '#9b59b6'
        }
        bar_colors = [colors.get(s, '#95a5a6') for s in splits]

        bars = ax.bar(splits, ginis, color=bar_colors, edgecolor='black', linewidth=1.2)

        # Add value labels on bars
        for bar, gini in zip(bars, ginis):
            height = bar.get_height()
            ax.annotate(
                f'{gini:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=12, fontweight='bold'
            )

        ax.set_xlabel('Data Split', fontsize=12)
        ax.set_ylabel('Gini', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(ginis) * 1.15 if ginis else 1)

        plt.tight_layout()
        return fig

    def plot_gini_by_groups(
        self,
        metrics: dict,
        split_name: str = 'valid',
        title: str = 'Gini by Groups',
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        '''
        Bar chart of Gini coefficient by group for a specific split.

        Args:
            metrics: Dictionary with split name -> metrics dict
            split_name: Which split to plot
            title: Plot title
            ax: Optional axes

        Returns:
            Matplotlib figure
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        split_metrics = metrics.get(split_name, {})
        gini_by_group = split_metrics.get('gini_by_group', {})

        if not gini_by_group:
            ax.text(0.5, 0.5, 'No group metrics available',
                    ha='center', va='center', transform=ax.transAxes)
            return fig

        groups = list(gini_by_group.keys())
        ginis = list(gini_by_group.values())

        # Sort by Gini descending
        sorted_idx = np.argsort(ginis)[::-1]
        groups = [groups[i] for i in sorted_idx]
        ginis = [ginis[i] for i in sorted_idx]

        bars = ax.barh(groups, ginis, color='steelblue', edgecolor='black')

        # Add value labels
        for bar, gini in zip(bars, ginis):
            width = bar.get_width()
            ax.annotate(
                f'{gini:.3f}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),
                textcoords='offset points',
                ha='left', va='center',
                fontsize=10
            )

        ax.set_xlabel('Gini', fontsize=12)
        ax.set_ylabel('Group', fontsize=12)
        ax.set_title(f'{title} ({split_name})', fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_gini_heatmap(
        self,
        metrics: dict,
        title: str = 'Gini by Groups and Splits'
    ) -> plt.Figure:
        '''
        Heatmap of Gini by groups (rows) and splits (columns).

        Args:
            metrics: Dictionary with split name -> metrics dict
            title: Plot title

        Returns:
            Matplotlib figure
        '''
        # Collect all groups
        all_groups = set()
        for split_metrics in metrics.values():
            if 'gini_by_group' in split_metrics:
                all_groups.update(split_metrics['gini_by_group'].keys())

        if not all_groups:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No group metrics available',
                    ha='center', va='center', transform=ax.transAxes)
            return fig

        # Build matrix
        splits = list(metrics.keys())
        groups = sorted(all_groups)

        data = np.zeros((len(groups), len(splits)))
        for j, split in enumerate(splits):
            gini_by_group = metrics[split].get('gini_by_group', {})
            for i, group in enumerate(groups):
                data[i, j] = gini_by_group.get(group, np.nan)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(8, len(splits) * 2), max(6, len(groups) * 0.5)))

        sns.heatmap(
            data,
            annot=True,
            fmt='.3f',
            xticklabels=splits,
            yticklabels=groups,
            cmap='RdYlGn',
            center=0.5,
            ax=ax,
            cbar_kws={'label': 'Gini'}
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Data Split', fontsize=12)
        ax.set_ylabel('Group', fontsize=12)

        plt.tight_layout()
        return fig

    def plot_metrics_summary(
        self,
        metrics: dict,
        title: str = 'Model Performance Summary'
    ) -> plt.Figure:
        '''
        Summary plot with Gini by splits and sample counts.

        Args:
            metrics: Dictionary with split name -> metrics dict
            title: Plot title

        Returns:
            Matplotlib figure
        '''
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Gini by splits
        self.plot_gini_by_splits(metrics, title='Gini by Split', ax=axes[0])

        # Sample counts
        splits = list(metrics.keys())
        counts = [metrics[s].get('n_samples', 0) for s in splits]
        target_rates = [metrics[s].get('target_rate', 0) for s in splits]

        colors = {
            'train': '#2ecc71',
            'valid': '#3498db',
            'oos': '#e74c3c',
            'oot': '#9b59b6'
        }
        bar_colors = [colors.get(s, '#95a5a6') for s in splits]

        bars = axes[1].bar(splits, counts, color=bar_colors, edgecolor='black', linewidth=1.2)

        # Add labels with count and target rate
        for bar, count, tr in zip(bars, counts, target_rates):
            height = bar.get_height()
            axes[1].annotate(
                f'{count:,}\n({tr:.1%})',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=10
            )

        axes[1].set_xlabel('Data Split', fontsize=12)
        axes[1].set_ylabel('Number of Samples', fontsize=12)
        axes[1].set_title('Sample Distribution', fontsize=14, fontweight='bold')

        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        title: str = 'Top Feature Importance'
    ) -> plt.Figure:
        '''
        Bar chart of top feature importance.

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to show
            title: Plot title

        Returns:
            Matplotlib figure
        '''
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))

        # Sort and take top N
        df = importance_df.nlargest(top_n, 'importance')

        ax.barh(df['feature'], df['importance'], color='steelblue', edgecolor='black')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        plt.tight_layout()
        return fig
