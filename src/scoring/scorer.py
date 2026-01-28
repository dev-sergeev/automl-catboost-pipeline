'Scorer for applying trained pipeline to new data.'

from typing import Optional

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.preprocessing import Preprocessor
from src.scoring.artifacts import ArtifactManager
from src.utils import get_logger


class Scorer:
    '''
    Score new data using trained pipeline artifacts.

    Loads artifacts and applies preprocessing + model to new data.
    '''

    def __init__(self, artifacts_dir: str = './artifacts'):
        self.artifacts_dir = artifacts_dir
        self.logger = get_logger('scoring')
        self.artifact_manager = ArtifactManager(artifacts_dir)

        self.model_ = None
        self.preprocessor_ = None
        self.features_: list[str] = []
        self.cat_features_: list[str] = []
        self.config_: Optional[dict] = None
        self.is_loaded_: bool = False

    def load(self) -> 'Scorer':
        'Load all artifacts for scoring.'
        self.logger.info(f'Loading scorer from {self.artifacts_dir}')

        artifacts = self.artifact_manager.load_all()

        # Load model
        self.model_ = artifacts['model']

        # Load features
        self.features_ = artifacts['features']
        self.cat_features_ = artifacts.get('cat_features', [])

        # Load config
        self.config_ = artifacts['config']

        # Reconstruct preprocessor
        config = PipelineConfig(**self.config_) if self.config_ else PipelineConfig()
        self.preprocessor_ = Preprocessor(config)
        self.preprocessor_.set_params(artifacts['preprocessor_params'])

        # Load balancer (not used for scoring, but available)
        self.balancer_params_ = artifacts.get('balancer_params', {})

        self.is_loaded_ = True
        self.logger.info(
            f'Scorer loaded: {len(self.features_)} features, '
            f'{len(self.cat_features_)} categorical'
        )

        return self

    def score(
        self,
        df: pd.DataFrame,
        return_proba: bool = True
    ) -> np.ndarray:
        '''
        Score new data.

        Args:
            df: Input DataFrame (raw data before preprocessing)
            return_proba: If True, return probabilities; if False, return classes

        Returns:
            Predictions (probabilities or classes)
        '''
        if not self.is_loaded_:
            self.load()

        self.logger.info(f'Scoring {len(df):,} rows...')

        # Apply preprocessing
        df_processed = self.preprocessor_.transform(df)

        # Select features
        available_features = [f for f in self.features_ if f in df_processed.columns]
        if len(available_features) < len(self.features_):
            missing = set(self.features_) - set(available_features)
            self.logger.warning(f'Missing features in input: {missing}')

        X = df_processed[available_features]

        # Score
        if return_proba:
            predictions = self.model_.predict_proba(X)[:, 1]
        else:
            predictions = self.model_.predict(X)

        self.logger.info('Scoring complete')
        return predictions

    def score_with_details(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        '''
        Score new data and return DataFrame with details.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with id columns and predictions
        '''
        if not self.is_loaded_:
            self.load()

        # Get id columns from config
        id_columns = self.config_.get('id_columns', [])
        if isinstance(id_columns, str):
            id_columns = [id_columns]

        # Score
        predictions = self.score(df)

        # Build result DataFrame
        result = df[id_columns].copy() if id_columns else pd.DataFrame()
        result['score'] = predictions

        return result

    def get_feature_importance(self) -> pd.DataFrame:
        'Get feature importance from the model.'
        if not self.is_loaded_:
            self.load()

        importance = self.model_.feature_importances_
        feature_names = self.model_.feature_names_

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        return df.sort_values('importance', ascending=False).reset_index(drop=True)

    def get_features(self) -> list[str]:
        'Get list of features used by the model.'
        if not self.is_loaded_:
            self.load()
        return self.features_.copy()

    def get_cat_features(self) -> list[str]:
        'Get list of categorical features.'
        if not self.is_loaded_:
            self.load()
        return self.cat_features_.copy()

    def get_metrics(self) -> dict:
        'Get metrics from training.'
        return self.artifact_manager.load_metrics()
