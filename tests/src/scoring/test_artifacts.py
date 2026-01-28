'Tests for artifact management.'

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier

from src.scoring.artifacts import ArtifactManager


class TestArtifactManager:
    'Tests for ArtifactManager.'

    def test_init(self, tmp_artifacts_dir):
        'Test artifact manager initialization.'
        manager = ArtifactManager(tmp_artifacts_dir)
        assert str(manager.artifacts_dir) == tmp_artifacts_dir

    def test_save_and_load_all(self, tmp_artifacts_dir, random_seed):
        'Test saving and loading all artifacts.'
        manager = ArtifactManager(tmp_artifacts_dir)

        # Create a simple model
        X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        y = pd.Series([0, 1, 0, 1, 0])

        model = CatBoostClassifier(iterations=10, verbose=0, random_seed=random_seed)
        model.fit(X, y)

        config = {'id_columns': ['client_id'], 'random_seed': random_seed}
        preprocessor_params = {'decimal_columns': [], 'date_columns': []}
        features = ['a', 'b']
        selector_params = {'selected_features': features}
        balancer_params = {'balance_columns': None}
        metrics = {'train': {'gini': 0.5}, 'valid': {'gini': 0.45}}

        # Save
        manager.save_all(
            config=config,
            preprocessor_params=preprocessor_params,
            features=features,
            selector_params=selector_params,
            model=model,
            balancer_params=balancer_params,
            metrics=metrics,
            cat_features=['b']
        )

        # Load
        artifacts = manager.load_all()

        assert artifacts['config'] == config
        assert artifacts['preprocessor_params'] == preprocessor_params
        assert artifacts['features'] == features
        assert artifacts['cat_features'] == ['b']
        assert artifacts['selector_params'] == selector_params
        assert artifacts['balancer_params'] == balancer_params
        assert artifacts['metrics'] == metrics
        assert artifacts['model'] is not None

    def test_load_model(self, tmp_artifacts_dir, random_seed):
        'Test loading only the model.'
        manager = ArtifactManager(tmp_artifacts_dir)

        X = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        y = pd.Series([0, 1, 0, 1, 0])

        model = CatBoostClassifier(iterations=10, verbose=0, random_seed=random_seed)
        model.fit(X, y)

        manager.save_all(
            config={},
            preprocessor_params={},
            features=['a'],
            selector_params={},
            model=model,
            balancer_params={},
            metrics={}
        )

        loaded_model = manager.load_model()

        # Check predictions are the same
        original_pred = model.predict_proba(X)[:, 1]
        loaded_pred = loaded_model.predict_proba(X)[:, 1]
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_load_features(self, tmp_artifacts_dir, random_seed):
        'Test loading features.'
        manager = ArtifactManager(tmp_artifacts_dir)

        X = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        y = pd.Series([0, 1, 0, 1, 0])

        model = CatBoostClassifier(iterations=10, verbose=0, random_seed=random_seed)
        model.fit(X, y)

        manager.save_all(
            config={},
            preprocessor_params={},
            features=['a', 'b', 'c'],
            selector_params={},
            model=model,
            balancer_params={},
            metrics={},
            cat_features=['c']
        )

        features, cat_features = manager.load_features()

        assert features == ['a', 'b', 'c']
        assert cat_features == ['c']

    def test_load_metrics(self, tmp_artifacts_dir, random_seed):
        'Test loading metrics.'
        manager = ArtifactManager(tmp_artifacts_dir)

        X = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        y = pd.Series([0, 1, 0, 1, 0])

        model = CatBoostClassifier(iterations=10, verbose=0, random_seed=random_seed)
        model.fit(X, y)

        metrics = {
            'train': {'gini': 0.6, 'auc': 0.8},
            'valid': {'gini': 0.55, 'auc': 0.775}
        }

        manager.save_all(
            config={},
            preprocessor_params={},
            features=['a'],
            selector_params={},
            model=model,
            balancer_params={},
            metrics=metrics
        )

        loaded_metrics = manager.load_metrics()

        assert loaded_metrics == metrics

    def test_exists(self, tmp_artifacts_dir, random_seed):
        'Test exists method.'
        manager = ArtifactManager(tmp_artifacts_dir)

        assert manager.exists() is False

        X = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        y = pd.Series([0, 1, 0, 1, 0])

        model = CatBoostClassifier(iterations=10, verbose=0, random_seed=random_seed)
        model.fit(X, y)

        manager.save_all(
            config={},
            preprocessor_params={},
            features=['a'],
            selector_params={},
            model=model,
            balancer_params={},
            metrics={}
        )

        assert manager.exists() is True

    def test_list_artifacts(self, tmp_artifacts_dir, random_seed):
        'Test listing artifact files.'
        manager = ArtifactManager(tmp_artifacts_dir)

        X = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        y = pd.Series([0, 1, 0, 1, 0])

        model = CatBoostClassifier(iterations=10, verbose=0, random_seed=random_seed)
        model.fit(X, y)

        manager.save_all(
            config={},
            preprocessor_params={},
            features=['a'],
            selector_params={},
            model=model,
            balancer_params={},
            metrics={}
        )

        files = manager.list_artifacts()

        assert 'config.json' in files
        assert 'model.cbm' in files
        assert 'features.json' in files
        assert 'metrics.json' in files

    def test_load_nonexistent_raises(self, tmp_artifacts_dir):
        'Test that loading from non-existent path raises error.'
        manager = ArtifactManager(tmp_artifacts_dir)

        with pytest.raises(FileNotFoundError):
            manager.load_all()
