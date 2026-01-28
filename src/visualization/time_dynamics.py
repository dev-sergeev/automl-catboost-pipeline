'Plots for metrics dynamics over time.'

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import get_logger


class TimeDynamicsPlotter:
    'Create plots for metrics dynamics over time.'

    def __init__(self, figsize: tuple[int, int] = (14, 6)):
        self.figsize = figsize
        self.logger = get_logger('visualization.time')

        plt.style.use('seaborn-v0_8-whitegrid')

    def plot_gini_over_time(
        self,
        time_metrics: pd.DataFrame,
        title: str = 'Gini Dynamics Over Time',
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        '''
        Line plot of Gini over time with sample count bars.

        Args:
            time_metrics: DataFrame with 'period', 'gini', 'n_samples' columns
            title: Plot title
            ax: Optional axes

        Returns:
            Matplotlib figure
        '''
        if ax is None:
            fig, ax1 = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
            ax1 = ax

        ax2 = ax1.twinx()

        periods = time_metrics['period']
        ginis = time_metrics['gini']
        counts = time_metrics['n_samples']

        # Bar chart for counts
        ax2.bar(
            periods, counts,
            alpha=0.3, color='gray', label='Sample Count'
        )

        # Line plot for Gini
        ax1.plot(
            periods, ginis,
            marker='o', linewidth=2, markersize=8,
            color='#e74c3c', label='Gini'
        )

        # Add Gini values on points
        for i, (_p, g) in enumerate(zip(periods, ginis)):
            if not np.isnan(g):
                ax1.annotate(
                    f'{g:.3f}',
                    xy=(i, g),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center', fontsize=9
                )

        ax1.set_xlabel('Period', fontsize=12)
        ax1.set_ylabel('Gini', fontsize=12, color='#e74c3c')
        ax2.set_ylabel('Sample Count', fontsize=12, color='gray')

        ax1.tick_params(axis='y', labelcolor='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='gray')

        # Rotate x labels if many periods
        if len(periods) > 6:
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax1.set_title(title, fontsize=14, fontweight='bold')

        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        return fig

    def plot_gini_by_groups_over_time(
        self,
        time_metrics_by_group: dict[str, pd.DataFrame],
        title: str = 'Gini Dynamics by Group'
    ) -> plt.Figure:
        '''
        Line plots of Gini over time for each group.

        Args:
            time_metrics_by_group: Dictionary group_name -> DataFrame
            title: Plot title

        Returns:
            Matplotlib figure
        '''
        fig, ax = plt.subplots(figsize=self.figsize)

        colors = plt.cm.tab10(np.linspace(0, 1, len(time_metrics_by_group)))

        for (group_name, df), color in zip(time_metrics_by_group.items(), colors):
            ax.plot(
                df['period'], df['gini'],
                marker='o', linewidth=2, markersize=6,
                label=group_name, color=color
            )

        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Gini', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)

        # Rotate x labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        return fig

    def plot_target_rate_over_time(
        self,
        time_metrics: pd.DataFrame,
        title: str = 'Target Rate Dynamics'
    ) -> plt.Figure:
        '''
        Line plot of target rate over time.

        Args:
            time_metrics: DataFrame with 'period', 'target_rate' columns
            title: Plot title

        Returns:
            Matplotlib figure
        '''
        fig, ax = plt.subplots(figsize=self.figsize)

        periods = time_metrics['period']
        target_rates = time_metrics['target_rate']

        ax.plot(
            periods, target_rates,
            marker='s', linewidth=2, markersize=8,
            color='#3498db'
        )

        # Fill area under curve
        ax.fill_between(periods, target_rates, alpha=0.3, color='#3498db')

        # Add values
        for i, (_p, tr) in enumerate(zip(periods, target_rates)):
            ax.annotate(
                f'{tr:.2%}',
                xy=(i, tr),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center', fontsize=9
            )

        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Target Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        return fig

    def plot_combined_time_analysis(
        self,
        time_metrics: pd.DataFrame,
        title: str = 'Model Performance Over Time'
    ) -> plt.Figure:
        '''
        Combined plot with Gini, target rate, and sample count.

        Args:
            time_metrics: DataFrame with time metrics
            title: Plot title

        Returns:
            Matplotlib figure
        '''
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        periods = time_metrics['period']

        # Gini
        axes[0].plot(
            periods, time_metrics['gini'],
            marker='o', linewidth=2, color='#e74c3c'
        )
        axes[0].set_ylabel('Gini', fontsize=12)
        axes[0].set_title('Gini Coefficient', fontsize=12, fontweight='bold')
        axes[0].axhline(y=time_metrics['gini'].mean(), color='gray',
                        linestyle='--', alpha=0.5, label='Mean')
        axes[0].legend()

        # Target rate
        axes[1].plot(
            periods, time_metrics['target_rate'],
            marker='s', linewidth=2, color='#3498db'
        )
        axes[1].set_ylabel('Target Rate', fontsize=12)
        axes[1].set_title('Target Rate', fontsize=12, fontweight='bold')
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        # Sample count
        axes[2].bar(
            periods, time_metrics['n_samples'],
            color='#2ecc71', alpha=0.7, edgecolor='black'
        )
        axes[2].set_ylabel('Sample Count', fontsize=12)
        axes[2].set_xlabel('Period', fontsize=12)
        axes[2].set_title('Sample Distribution', fontsize=12, fontweight='bold')

        # Rotate x labels
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_stability_analysis(
        self,
        time_metrics: pd.DataFrame,
        threshold: float = 0.05
    ) -> plt.Figure:
        '''
        Stability analysis showing Gini deviation from mean.

        Args:
            time_metrics: DataFrame with time metrics
            threshold: Threshold for highlighting unstable periods

        Returns:
            Matplotlib figure
        '''
        fig, ax = plt.subplots(figsize=self.figsize)

        periods = time_metrics['period']
        ginis = time_metrics['gini']
        mean_gini = ginis.mean()
        deviation = (ginis - mean_gini) / mean_gini

        # Color bars based on deviation
        colors = ['#e74c3c' if abs(d) > threshold else '#2ecc71' for d in deviation]

        ax.bar(periods, deviation, color=colors, edgecolor='black', alpha=0.7)

        # Add threshold lines
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label=f'+{threshold:.0%}')
        ax.axhline(y=-threshold, color='red', linestyle='--', alpha=0.5, label=f'-{threshold:.0%}')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Deviation from Mean', fontsize=12)
        ax.set_title(f'Gini Stability Analysis (Mean Gini: {mean_gini:.3f})',
                     fontsize=14, fontweight='bold')

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax.legend(loc='upper right')

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        return fig
