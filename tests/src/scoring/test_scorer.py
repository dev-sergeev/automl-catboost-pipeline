'Tests for scorer.'

import pandas as pd
import pytest
from catboost import CatBoostClassifier

from src.scoring import ArtifactManager, Scorer


class TestScorer:
    'Tests for Scorer.'

    @pytest.fixture
    def saved_artifacts(self, tmp_artifacts_dir, random_seed):
        'Create and save artifacts for testing.'
        manager = ArtifactManager(tmp_artifacts_dir)

        # Create simple training data
        X = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature_2': [5.0, 4.0, 3.0, 2.0, 1.0]
        })
        y = pd.Series([0, 1, 0, 1, 0])

        model = CatBoostClassifier(
            iterations=20,
            verbose=0,
            random_seed=random_seed
        )
        model.fit(X, y)

        config = {
            'id_columns': ['client_id'],
            'date_column': 'report_date',
            'target_column': 'target',
            'random_seed': random_seed
        }

        preprocessor_params = {
            'decimal_handler': {'decimal_columns': []},
            'date_transformer': {'report_date_column': 'report_date', 'date_columns': []},
            'type_detector': {
                'unique_threshold': 20,
                'exclude_columns': ['client_id', 'report_date', 'target'],
                'column_types': {
                    'feature_1': 'numeric',
                    'feature_2': 'numeric'
                },
                'numeric_columns': ['feature_1', 'feature_2'],
                'categorical_columns': []
            },
            'feature_columns': ['feature_1', 'feature_2']
        }

        features = ['feature_1', 'feature_2']
        selector_params = {'selected_features': features}
        balancer_params = {'balance_columns': None, 'group_weights': None, 'is_fitted': True}
        metrics = {'train': {'gini': 0.5, 'auc': 0.75}}

        manager.save_all(
            config=config,
            preprocessor_params=preprocessor_params,
            features=features,
            selector_params=selector_params,
            model=model,
            balancer_params=balancer_params,
            metrics=metrics
        )

        return tmp_artifacts_dir

    def test_init(self, tmp_artifacts_dir):
        'Test scorer initialization.'
        scorer = Scorer(tmp_artifacts_dir)
        assert scorer.artifacts_dir == tmp_artifacts_dir
        assert scorer.is_loaded_ is False

    def test_load(self, saved_artifacts):
        'Test loading artifacts.'
        scorer = Scorer(saved_artifacts)
        scorer.load()

        assert scorer.is_loaded_
        assert scorer.model_ is not None
        assert len(scorer.features_) == 2

    def test_score(self, saved_artifacts):
        'Test scoring new data.'
        scorer = Scorer(saved_artifacts)
        scorer.load()

        new_data = pd.DataFrame({
            'client_id': [1, 2, 3],
            'report_date': pd.date_range('2024-01-01', periods=3),
            'feature_1': [1.5, 2.5, 3.5],
            'feature_2': [4.5, 3.5, 2.5]
        })

        scores = scorer.score(new_data)

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_score_auto_load(self, saved_artifacts):
        'Test that score auto-loads if not loaded.'
        scorer = Scorer(saved_artifacts)

        new_data = pd.DataFrame({
            'client_id': [1, 2, 3],
            'report_date': pd.date_range('2024-01-01', periods=3),
            'feature_1': [1.5, 2.5, 3.5],
            'feature_2': [4.5, 3.5, 2.5]
        })

        scores = scorer.score(new_data)

        assert scorer.is_loaded_
        assert len(scores) == 3

    def test_score_with_details(self, saved_artifacts):
        'Test scoring with details.'
        scorer = Scorer(saved_artifacts)

        new_data = pd.DataFrame({
            'client_id': [1, 2, 3],
            'report_date': pd.date_range('2024-01-01', periods=3),
            'feature_1': [1.5, 2.5, 3.5],
            'feature_2': [4.5, 3.5, 2.5]
        })

        result = scorer.score_with_details(new_data)

        assert isinstance(result, pd.DataFrame)
        assert 'client_id' in result.columns
        assert 'score' in result.columns
        assert len(result) == 3

    def test_get_feature_importance(self, saved_artifacts):
        'Test getting feature importance.'
        scorer = Scorer(saved_artifacts)
        scorer.load()

        importance_df = scorer.get_feature_importance()

        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns

    def test_get_features(self, saved_artifacts):
        'Test getting features list.'
        scorer = Scorer(saved_artifacts)
        scorer.load()

        features = scorer.get_features()

        assert features == ['feature_1', 'feature_2']

    def test_get_metrics(self, saved_artifacts):
        'Test getting metrics.'
        scorer = Scorer(saved_artifacts)

        metrics = scorer.get_metrics()

        assert 'train' in metrics
        assert 'gini' in metrics['train']

    def test_return_classes(self, saved_artifacts):
        'Test returning class predictions.'
        scorer = Scorer(saved_artifacts)

        new_data = pd.DataFrame({
            'client_id': [1, 2, 3],
            'report_date': pd.date_range('2024-01-01', periods=3),
            'feature_1': [1.5, 2.5, 3.5],
            'feature_2': [4.5, 3.5, 2.5]
        })

        predictions = scorer.score(new_data, return_proba=False)

        assert len(predictions) == 3
        assert all(p in [0, 1] for p in predictions)
