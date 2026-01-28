'Tests for PSI filter.'

import numpy as np
import pandas as pd

from src.feature_selection.psi_filter import PSIFilter


class TestPSIFilter:
    'Tests for PSIFilter.'

    def test_init(self):
        'Test filter initialization.'
        filter_ = PSIFilter()
        assert filter_.threshold == 0.25
        assert filter_.n_bins == 10

        filter_ = PSIFilter(threshold=0.3, n_bins=20)
        assert filter_.threshold == 0.3
        assert filter_.n_bins == 20

    def test_stable_distribution_low_psi(self):
        'Test that stable distributions have low PSI.'
        np.random.seed(42)

        # Same distribution for reference and actual
        reference = pd.DataFrame({
            'feature': np.random.randn(1000)
        })
        actual = pd.DataFrame({
            'feature': np.random.randn(1000)
        })
        y = pd.Series(np.random.randint(0, 2, 1000))

        filter_ = PSIFilter(threshold=0.25)
        filter_.fit(actual, y, X_reference=reference)

        psi = filter_.psi_values_['feature']
        assert psi < 0.25  # Should be stable

    def test_shifted_distribution_high_psi(self):
        'Test that shifted distributions have high PSI.'
        np.random.seed(42)

        # Different distributions
        reference = pd.DataFrame({
            'feature': np.random.randn(1000)  # mean=0
        })
        actual = pd.DataFrame({
            'feature': np.random.randn(1000) + 3  # mean=3 (shifted)
        })
        y = pd.Series(np.random.randint(0, 2, 1000))

        filter_ = PSIFilter(threshold=0.25)
        filter_.fit(actual, y, X_reference=reference)

        psi = filter_.psi_values_['feature']
        assert psi > 0.25  # Should be unstable

    def test_removes_high_psi_features(self):
        'Test that high PSI features are removed.'
        np.random.seed(42)

        reference = pd.DataFrame({
            'stable': np.random.randn(1000),
            'unstable': np.random.randn(1000)
        })
        actual = pd.DataFrame({
            'stable': np.random.randn(1000),
            'unstable': np.random.randn(1000) + 5  # Shifted
        })
        y = pd.Series(np.random.randint(0, 2, 1000))

        filter_ = PSIFilter(threshold=0.25)
        filter_.fit(actual, y, X_reference=reference)

        assert 'stable' in filter_.selected_features_
        assert 'unstable' in filter_.removed_features_

    def test_categorical_psi(self):
        'Test PSI for categorical features.'
        np.random.seed(42)

        # Same category distribution
        reference = pd.DataFrame({
            'cat': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2])
        })
        actual = pd.DataFrame({
            'cat': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2])
        })
        y = pd.Series(np.random.randint(0, 2, 1000))

        filter_ = PSIFilter(threshold=0.25)
        filter_.fit(actual, y, X_reference=reference)

        psi = filter_.psi_values_['cat']
        assert psi < 0.25  # Should be stable

    def test_categorical_psi_shifted(self):
        'Test PSI for shifted categorical distributions.'
        np.random.seed(42)

        reference = pd.DataFrame({
            'cat': np.random.choice(['A', 'B', 'C'], 1000, p=[0.8, 0.1, 0.1])
        })
        actual = pd.DataFrame({
            'cat': np.random.choice(['A', 'B', 'C'], 1000, p=[0.1, 0.1, 0.8])  # Shifted
        })
        y = pd.Series(np.random.randint(0, 2, 1000))

        filter_ = PSIFilter(threshold=0.25)
        filter_.fit(actual, y, X_reference=reference)

        psi = filter_.psi_values_['cat']
        assert psi > 0.25  # Should be unstable

    def test_transform(self):
        'Test transform returns selected features.'
        np.random.seed(42)

        reference = pd.DataFrame({
            'stable': np.random.randn(1000),
            'unstable': np.random.randn(1000)
        })
        actual = pd.DataFrame({
            'stable': np.random.randn(1000),
            'unstable': np.random.randn(1000) + 5  # Large shift
        })
        y = pd.Series(np.random.randint(0, 2, 1000))

        filter_ = PSIFilter(threshold=0.25)
        filter_.fit(actual, y, X_reference=reference)

        result = filter_.transform(actual)

        # Unstable feature should be removed due to high PSI
        assert 'unstable' not in result.columns or filter_.psi_values_['unstable'] <= 0.25

    def test_get_psi_values(self):
        'Test getting PSI values.'
        np.random.seed(42)

        df = pd.DataFrame({
            'feature': np.random.randn(200)
        })
        y = pd.Series(np.random.randint(0, 2, 200))

        filter_ = PSIFilter()
        filter_.fit(df, y)  # Will split internally

        psi_values = filter_.get_psi_values()

        assert 'feature' in psi_values
        assert isinstance(psi_values['feature'], float)

    def test_get_set_params(self):
        'Test serialization.'
        np.random.seed(42)

        df = pd.DataFrame({
            'feature': np.random.randn(200)
        })
        y = pd.Series(np.random.randint(0, 2, 200))

        filter1 = PSIFilter(threshold=0.3)
        filter1.fit(df, y)
        params = filter1.get_params()

        filter2 = PSIFilter()
        filter2.set_params(params)

        assert filter2.threshold == filter1.threshold
        assert filter2.psi_values_ == filter1.psi_values_
        assert filter2.selected_features_ == filter1.selected_features_
