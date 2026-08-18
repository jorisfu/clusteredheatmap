import sys, os

from tests.tools import add_nans_uniform_everywhere

sys.path.append("./clusteredheatmap")
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
from clusteredheatmap.algos.distance import mesquita_eed
import pytest

from scipy.spatial.distance import pdist, squareform


@pytest.fixture
def sample_complete_data():
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.0, scale=1.0, size=(50, 25))


def test_complete_data_matches_euclidean(sample_complete_data):
    """When no NaNs are present, output must match standard Euclidean distance matrix."""
    X = sample_complete_data
    result = mesquita_eed(X)
    expected = pdist(X, metric="euclidean")

    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-8)


def test_fully_missing_vector_rejection():
    """Measurements without any observed values are not allowed"""
    X = np.array([[np.nan, np.nan], [1.0, 2.0]])

    with pytest.raises(ValueError):
        result = mesquita_eed(X)


def test_minimalnoise_matches_euclidean(sample_complete_data):
    """Adding minimal NaN noise shouldn't affect distance a lot"""
    rng = np.random.default_rng(123)
    X = sample_complete_data

    # Add some missingness (NaN noise)
    X_noisy = add_nans_uniform_everywhere(X, 0.01, rng)

    result = mesquita_eed(X_noisy)
    expected = pdist(X, metric="euclidean")

    assert result.shape == expected.shape
    np.testing.assert_allclose(
        squareform(result), squareform(expected), rtol=0.2
    )
