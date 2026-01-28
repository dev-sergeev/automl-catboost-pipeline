'Search space definitions for CatBoost hyperparameter optimization.'

from dataclasses import dataclass, field
from typing import Any

import optuna


@dataclass
class SearchSpace:
    'Definition of hyperparameter search space.'

    # Iterations
    iterations_min: int = 100
    iterations_max: int = 1000

    # Tree depth
    depth_min: int = 4
    depth_max: int = 10

    # Learning rate (log scale)
    learning_rate_min: float = 0.01
    learning_rate_max: float = 0.3

    # L2 regularization
    l2_leaf_reg_min: float = 1.0
    l2_leaf_reg_max: float = 10.0

    # Minimum samples in leaf
    min_data_in_leaf_min: int = 1
    min_data_in_leaf_max: int = 100

    # Border count (for numeric features)
    border_count_min: int = 32
    border_count_max: int = 255

    # Random strength
    random_strength_min: float = 0.0
    random_strength_max: float = 10.0

    # Bagging temperature
    bagging_temperature_min: float = 0.0
    bagging_temperature_max: float = 1.0

    # Grow policy options
    grow_policies: list[str] = field(
        default_factory=lambda: ['SymmetricTree', 'Depthwise', 'Lossguide']
    )

    # Bootstrap type options
    bootstrap_types: list[str] = field(
        default_factory=lambda: ['Bayesian', 'Bernoulli', 'MVS']
    )

    # Fixed parameters
    fixed_params: dict = field(default_factory=dict)

    def sample(self, trial: optuna.Trial) -> dict[str, Any]:
        'Sample hyperparameters from the search space.'
        params = {}

        # Iterations
        params['iterations'] = trial.suggest_int(
            'iterations',
            self.iterations_min,
            self.iterations_max
        )

        # Depth
        params['depth'] = trial.suggest_int(
            'depth',
            self.depth_min,
            self.depth_max
        )

        # Learning rate (log scale for better exploration)
        params['learning_rate'] = trial.suggest_float(
            'learning_rate',
            self.learning_rate_min,
            self.learning_rate_max,
            log=True
        )

        # L2 regularization
        params['l2_leaf_reg'] = trial.suggest_float(
            'l2_leaf_reg',
            self.l2_leaf_reg_min,
            self.l2_leaf_reg_max
        )

        # Min data in leaf
        params['min_data_in_leaf'] = trial.suggest_int(
            'min_data_in_leaf',
            self.min_data_in_leaf_min,
            self.min_data_in_leaf_max
        )

        # Grow policy
        params['grow_policy'] = trial.suggest_categorical(
            'grow_policy',
            self.grow_policies
        )

        # Border count (only for SymmetricTree)
        if params['grow_policy'] == 'SymmetricTree':
            params['border_count'] = trial.suggest_int(
                'border_count',
                self.border_count_min,
                self.border_count_max
            )

        # Bootstrap type
        bootstrap_type = trial.suggest_categorical(
            'bootstrap_type',
            self.bootstrap_types
        )
        params['bootstrap_type'] = bootstrap_type

        # Bootstrap-specific parameters
        if bootstrap_type == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float(
                'bagging_temperature',
                self.bagging_temperature_min,
                self.bagging_temperature_max
            )
        elif bootstrap_type == 'Bernoulli':
            params['subsample'] = trial.suggest_float(
                'subsample',
                0.5,
                1.0
            )

        # Random strength
        params['random_strength'] = trial.suggest_float(
            'random_strength',
            self.random_strength_min,
            self.random_strength_max
        )

        # Add fixed parameters
        params.update(self.fixed_params)

        return params


def get_default_search_space(
    task_type: str = 'CPU',
    include_grow_policy: bool = True,
    include_bootstrap: bool = True
) -> SearchSpace:
    'Get default search space with optional customization.'
    space = SearchSpace()

    if not include_grow_policy:
        space.grow_policies = ['SymmetricTree']

    if not include_bootstrap:
        space.bootstrap_types = ['Bayesian']

    if task_type == 'GPU':
        # GPU has some limitations
        space.grow_policies = ['SymmetricTree', 'Depthwise', 'Lossguide']
        # border_count is limited on GPU
        space.border_count_max = 128

    return space


def get_quick_search_space() -> SearchSpace:
    'Get a smaller search space for quick optimization.'
    return SearchSpace(
        iterations_min=100,
        iterations_max=500,
        depth_min=4,
        depth_max=8,
        learning_rate_min=0.03,
        learning_rate_max=0.2,
        l2_leaf_reg_min=1.0,
        l2_leaf_reg_max=5.0,
        min_data_in_leaf_min=1,
        min_data_in_leaf_max=50,
        grow_policies=['SymmetricTree'],
        bootstrap_types=['Bayesian']
    )
