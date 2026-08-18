from sklearn.datasets import make_blobs
import numpy as np

from clusteredheatmap.algos.modelselection import get_best_gmm
from tests.tools import add_nans_uniform_everywhere


def test_syntheticdata_lowdim():
    X, y = make_blobs(
        n_samples=100, centers=3, n_features=2, random_state=0, cluster_std=0.4
    )
    model = get_best_gmm(1, 5, 100, "AICc", X)
    assert model.n_components == 3
    model = get_best_gmm(1, 5, 100, "BIC", X)
    assert model.n_components == 3

    X, y = make_blobs(
        n_samples=100, centers=6, n_features=2, random_state=0, cluster_std=0.4
    )
    model = get_best_gmm(1, 10, 100, "AICc", X)
    assert model.n_components == 6
    model = get_best_gmm(1, 10, 100, "BIC", X)
    assert model.n_components == 6


# NOTE: On highdim data, the criteria prefer models with more components than the generator
def test_syntheticdata_highdim():
    X, y = make_blobs(
        n_samples=100, centers=3, n_features=70, random_state=0, cluster_std=0.4
    )
    model = get_best_gmm(1, 5, 100, "AICc", X)
    assert model.n_components == 3
    model = get_best_gmm(1, 5, 100, "BIC", X)
    assert model.n_components == 3

    X, y = make_blobs(
        n_samples=100, centers=6, n_features=70, random_state=0, cluster_std=0.4
    )
    model = get_best_gmm(1, 10, 100, "AICc", X)
    assert model.n_components == 6
    model = get_best_gmm(1, 10, 100, "BIC", X)
    assert model.n_components == 6


# NOTE: On noisy data, the criteria prefer models with more components than the generator
def test_syntheticdata_lowdim_noisy():
    rng = np.random.default_rng(123)

    X, y = make_blobs(
        n_samples=1000, centers=3, n_features=3, random_state=123, cluster_std=0.1
    )
    X_noisy = add_nans_uniform_everywhere(X, 0.01, rng)
    model = get_best_gmm(1, 10, 100, "AICc", X_noisy)
    assert model.n_components == 3
    model = get_best_gmm(1, 10, 100, "BIC", X_noisy)
    assert model.n_components == 3

    X, y = make_blobs(
        n_samples=1000, centers=6, n_features=3, random_state=123, cluster_std=0.1
    )
    X_noisy = add_nans_uniform_everywhere(X, 0.01, rng)
    model = get_best_gmm(1, 10, 100, "AICc", X_noisy)
    assert model.n_components == 6
    model = get_best_gmm(1, 10, 100, "BIC", X_noisy)
    assert model.n_components == 6


# NOTE: Here as well
def test_syntheticdata_highdim_noisy():
    rng = np.random.default_rng(123)

    X, y = make_blobs(
        n_samples=1000, centers=3, n_features=70, random_state=123, cluster_std=0.1
    )
    X_noisy = add_nans_uniform_everywhere(X, 0.01, rng)
    model = get_best_gmm(1, 10, 100, "AICc", X_noisy)
    assert model.n_components == 3
    model = get_best_gmm(1, 10, 100, "BIC", X_noisy)
    assert model.n_components == 3

    X, y = make_blobs(
        n_samples=1000, centers=6, n_features=70, random_state=123, cluster_std=0.1
    )
    X_noisy = add_nans_uniform_everywhere(X, 0.01, rng)
    model = get_best_gmm(1, 10, 100, "AICc", X_noisy)
    assert model.n_components == 6
    model = get_best_gmm(1, 10, 100, "BIC", X_noisy)
    assert model.n_components == 6
