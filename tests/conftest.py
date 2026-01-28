'Pytest fixtures for testing.'

import numpy as np
import pandas as pd
import pytest

from src.config import FeatureSelectionConfig, PipelineConfig


@pytest.fixture
def random_seed():
    'Fixed random seed for reproducibility.'
    return 42


@pytest.fixture
def sample_config(random_seed):
    'Sample pipeline configuration.'
    return PipelineConfig(
        id_columns=['client_id', 'app_id'],
        date_column='report_date',
        target_column='target',
        client_column='client_id',
        random_seed=random_seed,
        train_ratio=0.6,
        valid_ratio=0.2,
        oos_ratio=0.1,
        oot_ratio=0.1,
        n_trials=5,  # Few trials for testing
        optuna_timeout=60,
        catboost_iterations=50,  # Few iterations for testing
    )


@pytest.fixture
def sample_selection_config():
    'Sample feature selection configuration.'
    return FeatureSelectionConfig(
        run_missing_filter=True,
        run_variance_filter=True,
        run_correlation_filter=True,
        run_psi_filter=True,
        run_importance_filter=False,  # Skip slow steps in tests
        run_backward_selection=False,
        run_forward_selection=False,
    )


@pytest.fixture
def sample_df(random_seed):
    'Generate sample DataFrame for testing.'
    np.random.seed(random_seed)
    n_samples = 1000
    n_clients = 200

    # Generate client IDs (one client can have multiple observations)
    client_ids = np.random.choice(range(n_clients), size=n_samples)

    # Generate dates over 6 months
    dates = pd.date_range('2024-01-01', periods=180, freq='D')
    report_dates = np.random.choice(dates, size=n_samples)

    # Generate features
    data = {
        'client_id': client_ids,
        'app_id': range(n_samples),
        'report_date': report_dates,
        'product_type': np.random.choice(['PK', 'KK', 'IK'], size=n_samples, p=[0.8, 0.18, 0.02]),
        'num_feature_1': np.random.randn(n_samples),
        'num_feature_2': np.random.randn(n_samples) * 10 + 100,
        'num_feature_3': np.random.uniform(0, 1, n_samples),
        'num_feature_high_missing': np.where(
            np.random.random(n_samples) > 0.1, np.nan, np.random.randn(n_samples)
        ),
        'num_feature_zero_var': np.ones(n_samples) * 5,
        'cat_feature_1': np.random.choice(['A', 'B', 'C'], size=n_samples),
        'cat_feature_2': np.random.choice(['X', 'Y'], size=n_samples),
    }

    df = pd.DataFrame(data)

    # Generate target correlated with features
    logit = (
        0.5 * df['num_feature_1'] +
        0.01 * df['num_feature_2'] +
        2 * df['num_feature_3'] +
        np.where(df['cat_feature_1'] == 'A', 0.5, 0) +
        np.random.randn(n_samples) * 0.5
    )
    prob = 1 / (1 + np.exp(-logit))
    df['target'] = (np.random.random(n_samples) < prob).astype(int)

    return df


@pytest.fixture
def sample_df_small(random_seed):
    'Generate small sample DataFrame for quick tests.'
    np.random.seed(random_seed)
    n_samples = 100
    n_clients = 30

    client_ids = np.random.choice(range(n_clients), size=n_samples)
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    report_dates = np.random.choice(dates, size=n_samples)

    data = {
        'client_id': client_ids,
        'app_id': range(n_samples),
        'report_date': report_dates,
        'num_feature_1': np.random.randn(n_samples),
        'num_feature_2': np.random.randn(n_samples),
        'cat_feature_1': np.random.choice(['A', 'B'], size=n_samples),
    }

    df = pd.DataFrame(data)
    df['target'] = np.random.randint(0, 2, size=n_samples)

    return df


@pytest.fixture
def sample_train_valid_data(sample_df):
    'Split sample data into train and valid.'
    n = len(sample_df)
    train_idx = int(n * 0.8)

    train_df = sample_df.iloc[:train_idx].copy()
    valid_df = sample_df.iloc[train_idx:].copy()

    feature_cols = [c for c in sample_df.columns
                    if c not in ['client_id', 'app_id', 'report_date', 'target', 'product_type']]

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_valid = valid_df[feature_cols]
    y_valid = valid_df['target']

    return X_train, y_train, X_valid, y_valid


@pytest.fixture
def tmp_artifacts_dir(tmp_path):
    'Temporary directory for artifacts.'
    return str(tmp_path / 'artifacts')
