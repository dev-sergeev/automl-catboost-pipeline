'Optuna-based hyperparameter optimizer for CatBoost.'

from collections.abc import Callable
from typing import Any, Optional

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from src.config import OptunaConfig, PipelineConfig
from src.optimization.search_space import SearchSpace, get_default_search_space
from src.utils import get_logger


class OptunaOptimizer:
    'Optuna-based hyperparameter optimizer for CatBoost.'

    def __init__(
        self,
        config: PipelineConfig,
        optuna_config: Optional[OptunaConfig] = None,
        search_space: Optional[SearchSpace] = None
    ):
        self.config = config
        self.optuna_config = optuna_config or OptunaConfig(
            n_trials=config.n_trials,
            timeout=config.optuna_timeout
        )
        self.search_space = search_space or get_default_search_space(
            task_type=config.catboost_task_type
        )
        self.logger = get_logger('optimization')

        self.study_: Optional[optuna.Study] = None
        self.best_params_: dict = {}
        self.best_score_: float = 0.0
        self.is_fitted_: bool = False

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        cat_features: Optional[list[int]] = None
    ) -> dict[str, Any]:
        '''
        Run hyperparameter optimization.

        Args:
            X_train: Training features
            y_train: Training target
            X_valid: Validation features
            y_valid: Validation target
            sample_weight: Sample weights for training
            cat_features: Categorical feature indices

        Returns:
            Best hyperparameters
        '''
        self.logger.info(
            f'Starting Optuna optimization: {self.optuna_config.n_trials} trials, '
            f'timeout={self.optuna_config.timeout}s'
        )

        # Create objective function
        objective = self._create_objective(
            X_train, y_train, X_valid, y_valid,
            sample_weight=sample_weight,
            cat_features=cat_features
        )

        # Create study
        self.study_ = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.config.random_seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=self.optuna_config.pruning_warmup_steps,
                n_warmup_steps=self.optuna_config.pruning_interval_steps
            ) if self.optuna_config.use_pruning else None
        )

        # Run optimization
        self.study_.optimize(
            objective,
            n_trials=self.optuna_config.n_trials,
            timeout=self.optuna_config.timeout,
            n_jobs=self.optuna_config.n_jobs,
            show_progress_bar=self.optuna_config.show_progress_bar,
            callbacks=[self._logging_callback]
        )

        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value
        self.is_fitted_ = True

        self.logger.info(
            f'Optimization complete: best AUC={self.best_score_:.4f}'
        )
        self.logger.info(f'Best parameters: {self.best_params_}')

        return self.best_params_

    def _create_objective(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        sample_weight: Optional[np.ndarray],
        cat_features: Optional[list[int]]
    ) -> Callable[[optuna.Trial], float]:
        'Create Optuna objective function.'

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = self.search_space.sample(trial)

            # Add fixed parameters
            params['random_seed'] = self.config.random_seed
            params['verbose'] = 0
            params['thread_count'] = self.config.catboost_thread_count
            params['task_type'] = self.config.catboost_task_type
            params['eval_metric'] = 'AUC'
            params['early_stopping_rounds'] = self.config.catboost_early_stopping_rounds

            if cat_features:
                params['cat_features'] = cat_features

            # Create model
            model = CatBoostClassifier(**params)

            # Train with pruning callback if enabled
            if self.optuna_config.use_pruning:
                try:
                    from optuna.integration import CatBoostPruningCallback
                    pruning_callback = CatBoostPruningCallback(trial, 'AUC')
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_valid, y_valid),
                        sample_weight=sample_weight,
                        verbose=0,
                        callbacks=[pruning_callback]
                    )
                    pruning_callback.check_pruned()
                except (ImportError, ModuleNotFoundError):
                    # optuna-integration not installed, train without pruning
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_valid, y_valid),
                        sample_weight=sample_weight,
                        verbose=0
                    )
            else:
                model.fit(
                    X_train, y_train,
                    eval_set=(X_valid, y_valid),
                    sample_weight=sample_weight,
                    verbose=0
                )

            # Evaluate
            y_pred = model.predict_proba(X_valid)[:, 1]
            score = roc_auc_score(y_valid, y_pred)

            return score

        return objective

    def _logging_callback(self, study: optuna.Study, trial: 'optuna.trial.FrozenTrial') -> None:
        'Callback for logging progress.'
        if trial.number % 10 == 0:
            self.logger.info(
                f'Trial {trial.number}: AUC={trial.value:.4f}, '
                f'best so far={study.best_value:.4f}'
            )

    def get_best_params(self) -> dict[str, Any]:
        'Get best hyperparameters.'
        if not self.is_fitted_:
            raise RuntimeError('Optimizer must be fitted first')
        return self.best_params_.copy()

    def get_best_score(self) -> float:
        'Get best score achieved.'
        if not self.is_fitted_:
            raise RuntimeError('Optimizer must be fitted first')
        return self.best_score_

    def get_study(self) -> optuna.Study:
        'Get the Optuna study object.'
        if self.study_ is None:
            raise RuntimeError('Optimizer must be fitted first')
        return self.study_

    def get_trials_dataframe(self) -> pd.DataFrame:
        'Get trials as DataFrame for analysis.'
        if self.study_ is None:
            raise RuntimeError('Optimizer must be fitted first')
        return self.study_.trials_dataframe()

    def get_importance(self) -> dict[str, float]:
        'Get hyperparameter importance.'
        if self.study_ is None:
            raise RuntimeError('Optimizer must be fitted first')
        return optuna.importance.get_param_importances(self.study_)

    def get_params(self) -> dict:
        'Get optimizer state for serialization.'
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'is_fitted': self.is_fitted_
        }

    def set_params(self, params: dict) -> 'OptunaOptimizer':
        'Set optimizer state from serialization.'
        self.best_params_ = params.get('best_params', {})
        self.best_score_ = params.get('best_score', 0.0)
        self.is_fitted_ = params.get('is_fitted', False)
        return self
