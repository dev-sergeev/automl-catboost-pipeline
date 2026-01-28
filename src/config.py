'Pipeline configuration using dataclasses.'

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class PipelineConfig:
    'Configuration for the CatBoost scoring pipeline.'

    # Identifiers
    id_columns: Union[str, list[str]] = field(default_factory=lambda: ['client_id'])
    date_column: str = 'report_date'
    target_column: str = 'target'
    random_seed: int = 42

    # Client column for group-aware split (defaults to first id_column)
    client_column: Optional[str] = None

    # Balance columns for sample weighting (e.g., product types)
    balance_columns: Optional[Union[str, list[str]]] = None

    # Data splitting ratios
    train_ratio: float = 0.6
    valid_ratio: float = 0.2
    oos_ratio: float = 0.1  # Out-of-sample
    oot_ratio: float = 0.1  # Out-of-time

    # Preprocessing
    unique_threshold: int = 20  # Threshold for numeric vs category

    # Feature selection thresholds
    missing_threshold: float = 0.95
    variance_threshold: float = 0.0
    correlation_threshold: float = 0.95
    psi_threshold: float = 0.25
    importance_threshold: float = 0.0
    backward_drop_ratio: float = 0.10  # 10% per iteration
    forward_add_ratio: float = 0.10

    # Optuna settings
    n_trials: int = 100
    optuna_timeout: int = 3600  # 1 hour

    # CatBoost settings
    catboost_iterations: int = 1000
    catboost_early_stopping_rounds: int = 50
    catboost_thread_count: int = -1
    catboost_task_type: str = 'CPU'  # 'CPU' or 'GPU'
    catboost_verbose: int = 0

    # Artifacts
    artifacts_dir: str = './artifacts'

    def __post_init__(self):
        'Validate and normalize configuration.'
        # Normalize id_columns to list
        if isinstance(self.id_columns, str):
            self.id_columns = [self.id_columns]

        # Set client_column to first id_column if not specified
        if self.client_column is None:
            self.client_column = self.id_columns[0]

        # Normalize balance_columns to list if specified
        if isinstance(self.balance_columns, str):
            self.balance_columns = [self.balance_columns]

        # Validate ratios
        total_ratio = self.train_ratio + self.valid_ratio + self.oos_ratio + self.oot_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f'Split ratios must sum to 1.0, got {total_ratio}'
            )

        # Validate thresholds
        if not 0 <= self.missing_threshold <= 1:
            raise ValueError('missing_threshold must be between 0 and 1')
        if not 0 <= self.correlation_threshold <= 1:
            raise ValueError('correlation_threshold must be between 0 and 1')
        if self.psi_threshold < 0:
            raise ValueError('psi_threshold must be non-negative')

    def get_id_columns_list(self) -> list[str]:
        'Return id_columns as a list.'
        if isinstance(self.id_columns, str):
            return [self.id_columns]
        return self.id_columns

    def get_balance_columns_list(self) -> Optional[list[str]]:
        'Return balance_columns as a list or None.'
        if self.balance_columns is None:
            return None
        if isinstance(self.balance_columns, str):
            return [self.balance_columns]
        return self.balance_columns


@dataclass
class FeatureSelectionConfig:
    'Configuration for feature selection steps.'

    # Which steps to run
    run_missing_filter: bool = True
    run_variance_filter: bool = True
    run_correlation_filter: bool = True
    run_psi_filter: bool = True
    run_psi_time_filter: bool = True  # PSI stability over time
    run_importance_filter: bool = True
    run_backward_selection: bool = False  # Optional, slower
    run_forward_selection: bool = False   # Optional, slowest

    # Thresholds (inherited from PipelineConfig if not overridden)
    missing_threshold: Optional[float] = None
    variance_threshold: Optional[float] = None
    correlation_threshold: Optional[float] = None
    psi_threshold: Optional[float] = None
    importance_threshold: Optional[float] = None
    backward_drop_ratio: Optional[float] = None
    forward_add_ratio: Optional[float] = None

    # PSI time filter settings
    psi_time_period_days: int = 14  # Days per period for time-based PSI
    psi_time_min_period_days: int = 14  # Minimum days in last period

    # Backward/Forward selection settings
    backward_min_features: int = 10
    forward_max_features: int = 100


@dataclass
class OptunaConfig:
    'Configuration for Optuna hyperparameter optimization.'

    n_trials: int = 100
    timeout: int = 3600
    n_jobs: int = 1
    show_progress_bar: bool = True

    # Search space bounds
    iterations_min: int = 100
    iterations_max: int = 1000
    depth_min: int = 4
    depth_max: int = 10
    learning_rate_min: float = 0.01
    learning_rate_max: float = 0.3
    l2_leaf_reg_min: float = 1.0
    l2_leaf_reg_max: float = 10.0
    min_data_in_leaf_min: int = 1
    min_data_in_leaf_max: int = 100

    # Pruning
    use_pruning: bool = True
    pruning_warmup_steps: int = 10
    pruning_interval_steps: int = 5
