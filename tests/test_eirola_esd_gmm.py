import sys, os

sys.path.append("./clusteredheatmap")
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
from clusteredheatmap.algos.distance import eirola_esd_gmm
import pytest

from scipy.spatial.distance import pdist, squareform


@pytest.fixture
def sample_complete_data():
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.0, scale=1.0, size=(10, 5))


def test_complete_data_matches_sqeuclidean(sample_complete_data):
    """When no NaNs are present, output must match standard squared Euclidean distance matrix."""
    X = sample_complete_data
    result = eirola_esd_gmm(X)
    expected = pdist(X, metric="sqeuclidean")

    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-8)


def test_fully_missing_vector_rejection():
    """Measurements without any observed values are not allowed"""
    X = np.array([[np.nan, np.nan], [1.0, 2.0]])

    with pytest.raises(ValueError):
        result = eirola_esd_gmm(X)


def test_minimalnoise_matches_sqeuclidean(sample_complete_data):
    """Adding minimal NaN noise shouldn't affect distance a lot"""
    rng = np.random.default_rng(123)
    X = sample_complete_data

    # Add some missingness (NaN noise)
    X_noisy = X.copy()
    X_noisy[0, 4] = np.nan

    result = eirola_esd_gmm(X_noisy)
    expected = pdist(X, metric="sqeuclidean")

    assert result.shape == expected.shape
    np.testing.assert_allclose(
        squareform(result), squareform(expected), rtol=1e-6, atol=1e-8
    )
