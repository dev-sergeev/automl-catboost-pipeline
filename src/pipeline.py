'Main CatBoost pipeline orchestrator.'

from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.config import FeatureSelectionConfig, PipelineConfig
from src.data import DataSplitter, SampleBalancer, SplitResult
from src.evaluation import MetricsCalculator
from src.feature_selection import FeatureSelector
from src.optimization import OptunaOptimizer
from src.preprocessing import Preprocessor
from src.scoring import ArtifactManager, Scorer
from src.utils import get_logger, set_random_seed
from src.visualization import MetricsPlotter, TimeDynamicsPlotter


class CatBoostPipeline:
    '''
    Main pipeline for training CatBoost scoring models.

    Orchestrates:
    1. Data loading and splitting
    2. Sample balancing (optional)
    3. Preprocessing
    4. Feature selection
    5. Hyperparameter optimization
    6. Model training
    7. Evaluation
    8. Artifact saving
    '''

    def __init__(
        self,
        config: PipelineConfig,
        selection_config: Optional[FeatureSelectionConfig] = None
    ):
        self.config = config
        self.selection_config = selection_config or FeatureSelectionConfig()
        self.logger = get_logger('pipeline')

        # Initialize components
        self.data_splitter = DataSplitter(config)
        self.balancer = SampleBalancer(config)
        self.preprocessor = Preprocessor(config)
        self.selector = FeatureSelector(config, self.selection_config)
        self.optimizer = OptunaOptimizer(config)
        self.metrics_calculator = MetricsCalculator()
        self.artifact_manager = ArtifactManager(config.artifacts_dir)

        # State
        self.model_: Optional[CatBoostClassifier] = None
        self.split_result_: Optional[SplitResult] = None
        self.sample_weights_: Optional[np.ndarray] = None
        self.metrics_: dict = {}
        self.is_fitted_: bool = False

        # Set random seed
        set_random_seed(config.random_seed)

    def fit(
        self,
        df: pd.DataFrame,
        run_optimization: bool = True,
        save_artifacts: bool = True
    ) -> 'CatBoostPipeline':
        '''
        Fit the complete pipeline.

        Args:
            df: Input DataFrame with features, target, and ids
            run_optimization: Whether to run hyperparameter optimization
            save_artifacts: Whether to save artifacts after training

        Returns:
            self
        '''
        self.logger.info(f'Starting pipeline on {len(df):,} rows, {len(df.columns)} columns')

        # 1. Split data
        self.logger.info('Step 1: Splitting data...')
        self.split_result_ = self.data_splitter.split(df)

        # 2. Fit balancer on training data
        self.logger.info('Step 2: Computing sample weights...')
        self.balancer.fit(self.split_result_.train)
        self.sample_weights_ = self.balancer.transform(self.split_result_.train)

        # 3. Preprocess
        self.logger.info('Step 3: Preprocessing...')
        self.preprocessor.fit(self.split_result_.train)

        train_processed = self.preprocessor.transform(self.split_result_.train)
        valid_processed = self.preprocessor.transform(self.split_result_.valid)

        # 4. Prepare features
        feature_columns = self.preprocessor.get_feature_columns()
        cat_features = self.preprocessor.get_categorical_features()

        X_train = train_processed[feature_columns]
        y_train = self.split_result_.train[self.config.target_column]
        X_valid = valid_processed[feature_columns]
        y_valid = self.split_result_.valid[self.config.target_column]

        # 5. Feature selection
        self.logger.info('Step 4: Feature selection...')
        self.selector.fit(
            X_train, y_train, X_valid, y_valid,
            sample_weight=self.sample_weights_,
            cat_features=cat_features,
            date_column=self.config.date_column,
            train_dates=self.split_result_.train[self.config.date_column],
            valid_dates=self.split_result_.valid[self.config.date_column]
        )

        selected_features = self.selector.get_selected_features()
        selected_cat_features = [f for f in cat_features if f in selected_features]

        X_train_selected = X_train[selected_features]
        X_valid_selected = X_valid[selected_features]

        # Get cat feature indices
        cat_feature_indices = [
            i for i, f in enumerate(selected_features)
            if f in selected_cat_features
        ]

        # 6. Hyperparameter optimization
        if run_optimization:
            self.logger.info('Step 5: Hyperparameter optimization...')
            best_params = self.optimizer.optimize(
                X_train_selected, y_train,
                X_valid_selected, y_valid,
                sample_weight=self.sample_weights_,
                cat_features=cat_feature_indices
            )
        else:
            best_params = {
                'iterations': self.config.catboost_iterations,
                'depth': 6,
                'learning_rate': 0.1
            }

        # 7. Train final model
        self.logger.info('Step 6: Training final model...')
        self.model_ = self._train_final_model(
            X_train_selected, y_train,
            X_valid_selected, y_valid,
            best_params,
            cat_feature_indices
        )

        # 8. Evaluate on all splits
        self.logger.info('Step 7: Evaluating model...')
        self.metrics_ = self._evaluate_all_splits(selected_features)

        self.is_fitted_ = True

        # 9. Save artifacts
        if save_artifacts:
            self.logger.info('Step 8: Saving artifacts...')
            self._save_artifacts(selected_features, selected_cat_features)

        self.logger.info('Pipeline complete!')
        self._log_summary()

        return self

    def _train_final_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        params: dict,
        cat_features: list[int]
    ) -> CatBoostClassifier:
        'Train the final CatBoost model.'
        model_params = {
            'random_seed': self.config.random_seed,
            'verbose': self.config.catboost_verbose,
            'thread_count': self.config.catboost_thread_count,
            'task_type': self.config.catboost_task_type,
            'eval_metric': 'AUC',
            'early_stopping_rounds': self.config.catboost_early_stopping_rounds,
            'cat_features': cat_features
        }
        model_params.update(params)

        model = CatBoostClassifier(**model_params)

        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            sample_weight=self.sample_weights_,
            verbose=self.config.catboost_verbose
        )

        return model

    def _evaluate_all_splits(self, selected_features: list[str]) -> dict:
        'Evaluate model on all data splits.'
        metrics = {}
        balance_cols = self.config.get_balance_columns_list()

        for split_name in ['train', 'valid', 'oos', 'oot']:
            split_df = self.split_result_.get_split(split_name)
            split_processed = self.preprocessor.transform(split_df)

            X = split_processed[selected_features]
            y_true = split_df[self.config.target_column].values
            y_pred = self.model_.predict_proba(X)[:, 1]

            # Overall metrics
            split_metrics = self.metrics_calculator.calculate_metrics(y_true, y_pred)

            # Metrics by group if balance_columns specified
            if balance_cols:
                group_key = self._get_group_key(split_df, balance_cols)
                gini_by_group = self.metrics_calculator.calculate_metrics_by_group(
                    y_true, y_pred, group_key
                )
                split_metrics['gini_by_group'] = {
                    g: m.get('gini', np.nan) for g, m in gini_by_group.items()
                }

            metrics[split_name] = split_metrics

        return metrics

    def _get_group_key(self, df: pd.DataFrame, balance_cols: list[str]) -> pd.Series:
        'Create group key from balance columns.'
        if len(balance_cols) == 1:
            return df[balance_cols[0]].astype(str)
        return df[balance_cols].astype(str).agg('|'.join, axis=1)

    def _save_artifacts(
        self,
        features: list[str],
        cat_features: list[str]
    ) -> None:
        'Save all pipeline artifacts.'
        self.artifact_manager.save_all(
            config=asdict(self.config),
            preprocessor_params=self.preprocessor.get_params(),
            features=features,
            selector_params=self.selector.get_params(),
            model=self.model_,
            balancer_params=self.balancer.get_params(),
            metrics=self.metrics_,
            optimizer_params=self.optimizer.get_params() if self.optimizer.is_fitted_ else None,
            cat_features=cat_features
        )

    def _log_summary(self) -> None:
        'Log training summary.'
        self.logger.info('=' * 60)
        self.logger.info('TRAINING SUMMARY')
        self.logger.info('=' * 60)

        for split_name, split_metrics in self.metrics_.items():
            gini = split_metrics.get('gini', 0)
            auc = split_metrics.get('auc', 0)
            n = split_metrics.get('n_samples', 0)
            self.logger.info(
                f'{split_name.upper():6s}: Gini={gini:.4f}, AUC={auc:.4f}, N={n:,}'
            )

        self.logger.info('=' * 60)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        'Predict probabilities for new data.'
        if not self.is_fitted_:
            raise RuntimeError('Pipeline must be fitted before predict')

        processed = self.preprocessor.transform(df)
        features = self.selector.get_selected_features()
        X = processed[features]
        return self.model_.predict_proba(X)[:, 1]

    def get_metrics(self) -> dict:
        'Get evaluation metrics.'
        return self.metrics_.copy()

    def get_feature_importance(self) -> pd.DataFrame:
        'Get feature importance from the model.'
        if not self.is_fitted_:
            raise RuntimeError('Pipeline must be fitted first')

        importance = self.model_.feature_importances_
        feature_names = self.model_.feature_names_

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        return df.sort_values('importance', ascending=False).reset_index(drop=True)

    def plot_metrics(self):
        'Create metrics visualization.'
        plotter = MetricsPlotter()
        return plotter.plot_metrics_summary(self.metrics_)

    def plot_feature_importance(self, top_n: int = 20):
        'Create feature importance plot.'
        plotter = MetricsPlotter()
        importance_df = self.get_feature_importance()
        return plotter.plot_feature_importance(importance_df, top_n=top_n)

    def calculate_time_metrics(self, split_name: str = 'oot') -> pd.DataFrame:
        'Calculate metrics over time for a split.'
        if not self.is_fitted_:
            raise RuntimeError('Pipeline must be fitted first')

        split_df = self.split_result_.get_split(split_name)
        processed = self.preprocessor.transform(split_df)
        features = self.selector.get_selected_features()

        X = processed[features]
        y_true = split_df[self.config.target_column].values
        y_pred = self.model_.predict_proba(X)[:, 1]
        dates = split_df[self.config.date_column]

        return self.metrics_calculator.calculate_metrics_by_time(y_true, y_pred, dates)

    def plot_time_dynamics(self, split_name: str = 'oot'):
        'Create time dynamics plot.'
        time_metrics = self.calculate_time_metrics(split_name)
        plotter = TimeDynamicsPlotter()
        return plotter.plot_gini_over_time(time_metrics)

    @classmethod
    def load(cls, artifacts_dir: str) -> 'Scorer':
        'Load a trained pipeline as Scorer for inference.'
        scorer = Scorer(artifacts_dir)
        scorer.load()
        return scorer

    @property
    def metrics(self) -> dict:
        'Alias for get_metrics().'
        return self.get_metrics()
