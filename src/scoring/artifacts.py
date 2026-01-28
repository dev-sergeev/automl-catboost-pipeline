'Artifact management for model persistence.'

import json
from pathlib import Path
from typing import Any, Optional

import joblib

from src.utils import get_logger


class ArtifactManager:
    '''
    Manage saving and loading of pipeline artifacts.

    Artifacts:
    - config.json: Pipeline configuration
    - preprocessor.pkl: Preprocessing parameters
    - features.json: Selected features list
    - selector.pkl: Feature selector parameters
    - model.cbm: CatBoost model
    - balancer.pkl: Sample balancer parameters
    - metrics.json: Evaluation metrics
    - optimizer.pkl: Optuna optimizer state
    '''

    def __init__(self, artifacts_dir: str = './artifacts'):
        self.artifacts_dir = Path(artifacts_dir)
        self.logger = get_logger('scoring.artifacts')

    def save_all(
        self,
        config: dict,
        preprocessor_params: dict,
        features: list[str],
        selector_params: dict,
        model,
        balancer_params: dict,
        metrics: dict,
        optimizer_params: Optional[dict] = None,
        cat_features: Optional[list[str]] = None
    ) -> None:
        'Save all pipeline artifacts.'
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f'Saving artifacts to {self.artifacts_dir}')

        # Save config
        self._save_json(config, 'config.json')

        # Save preprocessor
        self._save_pickle(preprocessor_params, 'preprocessor.pkl')

        # Save features
        self._save_json({
            'features': features,
            'cat_features': cat_features or []
        }, 'features.json')

        # Save selector
        self._save_pickle(selector_params, 'selector.pkl')

        # Save model
        model_path = self.artifacts_dir / 'model.cbm'
        model.save_model(str(model_path))
        self.logger.info(f'Saved model to {model_path}')

        # Save balancer
        self._save_pickle(balancer_params, 'balancer.pkl')

        # Save metrics
        self._save_json(metrics, 'metrics.json')

        # Save optimizer if provided
        if optimizer_params:
            self._save_pickle(optimizer_params, 'optimizer.pkl')

        self.logger.info('All artifacts saved successfully')

    def load_all(self) -> dict[str, Any]:
        'Load all pipeline artifacts.'
        if not self.artifacts_dir.exists():
            raise FileNotFoundError(
                f'Artifacts directory not found: {self.artifacts_dir}'
            )

        self.logger.info(f'Loading artifacts from {self.artifacts_dir}')

        artifacts = {}

        # Load config
        artifacts['config'] = self._load_json('config.json')

        # Load preprocessor
        artifacts['preprocessor_params'] = self._load_pickle('preprocessor.pkl')

        # Load features
        features_data = self._load_json('features.json')
        artifacts['features'] = features_data['features']
        artifacts['cat_features'] = features_data.get('cat_features', [])

        # Load selector
        artifacts['selector_params'] = self._load_pickle('selector.pkl')

        # Load model
        from catboost import CatBoostClassifier
        model_path = self.artifacts_dir / 'model.cbm'
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        artifacts['model'] = model
        self.logger.info(f'Loaded model from {model_path}')

        # Load balancer
        artifacts['balancer_params'] = self._load_pickle('balancer.pkl')

        # Load metrics
        artifacts['metrics'] = self._load_json('metrics.json')

        # Load optimizer if exists
        optimizer_path = self.artifacts_dir / 'optimizer.pkl'
        if optimizer_path.exists():
            artifacts['optimizer_params'] = self._load_pickle('optimizer.pkl')

        self.logger.info('All artifacts loaded successfully')
        return artifacts

    def load_model(self):
        'Load only the CatBoost model.'
        from catboost import CatBoostClassifier
        model_path = self.artifacts_dir / 'model.cbm'
        if not model_path.exists():
            raise FileNotFoundError(f'Model not found: {model_path}')
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        return model

    def load_features(self) -> tuple[list[str], list[str]]:
        'Load feature list and categorical features.'
        features_data = self._load_json('features.json')
        return features_data['features'], features_data.get('cat_features', [])

    def load_preprocessor_params(self) -> dict:
        'Load preprocessor parameters.'
        return self._load_pickle('preprocessor.pkl')

    def load_balancer_params(self) -> dict:
        'Load balancer parameters.'
        return self._load_pickle('balancer.pkl')

    def load_metrics(self) -> dict:
        'Load evaluation metrics.'
        return self._load_json('metrics.json')

    def load_config(self) -> dict:
        'Load pipeline configuration.'
        return self._load_json('config.json')

    def _save_json(self, data: Any, filename: str) -> None:
        'Save data as JSON.'
        path = self.artifacts_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        self.logger.debug(f'Saved {filename}')

    def _load_json(self, filename: str) -> Any:
        'Load data from JSON.'
        path = self.artifacts_dir / filename
        if not path.exists():
            raise FileNotFoundError(f'Artifact not found: {path}')
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def _save_pickle(self, data: Any, filename: str) -> None:
        'Save data as pickle.'
        path = self.artifacts_dir / filename
        joblib.dump(data, path)
        self.logger.debug(f'Saved {filename}')

    def _load_pickle(self, filename: str) -> Any:
        'Load data from pickle.'
        path = self.artifacts_dir / filename
        if not path.exists():
            raise FileNotFoundError(f'Artifact not found: {path}')
        return joblib.load(path)

    def exists(self) -> bool:
        'Check if artifacts directory exists with model.'
        model_path = self.artifacts_dir / 'model.cbm'
        return model_path.exists()

    def list_artifacts(self) -> list[str]:
        'List all artifact files.'
        if not self.artifacts_dir.exists():
            return []
        return [f.name for f in self.artifacts_dir.iterdir() if f.is_file()]
