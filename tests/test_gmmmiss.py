import sys, os

sys.path.append("./clusteredheatmap")
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np

from clusteredheatmap.algos.gmm_missing.gmm import GMMMissing
from sklearn.mixture import GaussianMixture


def test_minimal_nomissing_novariance():
    """
    All vectors are the same, no NaNs.
    Should have obvious means and no covariance (except for the reg_covar adjustment)
    """
    data = np.array(
        [
            [7.0, 56.0, 10.0],
            [7.0, 56.0, 10.0],
            [7.0, 56.0, 10.0],
            [7.0, 56.0, 10.0],
        ]
    )
    g = GMMMissing(n_components=1, max_iter=400, reg_covar=1e-6).fit(data)
    assert g.mu_ is not None
    assert g.covariances_ is not None
    assert g.mu_.shape == (1, 3)
    np.testing.assert_allclose(g.mu_[0], [7.0, 56.0, 10.0], rtol=1e-6, atol=1e-8)
    expected_cov = [
        [1.0e-06, 0.0e00, 0.0e00],
        [0.0e00, 1.0e-06, 0.0e00],
        [0.0e00, 0.0e00, 1.0e-06],
    ]
    np.testing.assert_allclose(g.covariances_[0], expected_cov, rtol=1e-6, atol=1e-8)


def test_minimal_missing():
    """
    All vectors are the same, with some NaNs.
    Should still have obvious means and no covariance (except for the reg_covar adjustment)
    """
    data = np.array(
        [
            [np.nan, 56.0, 10.0],
            [7.0, 56.0, np.nan],
            [7.0, np.nan, 10.0],
            [7.0, 56.0, 10.0],
        ]
    )
    g = GMMMissing(n_components=1, max_iter=400, reg_covar=1e-6).fit(data)
    assert g.mu_ is not None
    assert g.covariances_ is not None
    assert g.mu_.shape == (1, 3)
    np.testing.assert_allclose(g.mu_[0], [7.0, 56.0, 10.0], rtol=1e-6, atol=1e-8)
    expected_cov = [
        [1.0e-06, 0.0e00, 0.0e00],
        [0.0e00, 1.0e-06, 0.0e00],
        [0.0e00, 0.0e00, 1.0e-06],
    ]
    np.testing.assert_allclose(g.covariances_[0], expected_cov, rtol=1e-6, atol=1e-8)


def test_minimal_missing_somevariance_means():
    """
    Vectors have some variance and NaNs, shouldn't break model.
    """
    data = np.array(
        [
            [5.1, 10.2, np.nan],
            [4.9, np.nan, np.nan],
            [5.0, 9.9, 1.0],
            [np.nan, 10.1, 0.9],
            [5.2, 10.5, 1.1],
            [np.nan, 11.0, 1.2],
        ]
    )
    g = GMMMissing(n_components=1, max_iter=400, reg_covar=1e-6).fit(data)
    assert g.mu_ is not None
    np.testing.assert_allclose(g.mu_[0], [5.1, 10.0, 1.0], rtol=0.05)


def test_minimal_missing_explicit_covariance():
    """
    Vectors have some variance and NaNs, shouldn't break model.
    Also we have some more covariance between feature 0 and 1 (feat1 = 4*feat0)
    """
    data = np.array(
        [
            [5.1, 20.4, np.nan],
            [4.9, np.nan, np.nan],
            [5.0, 20.0, 1.0],
            [np.nan, 20.1, 0.9],
            [5.2, 20.8, 1.1],
            [np.nan, 22.0, 1.2],
        ]
    )
    g = GMMMissing(n_components=1, max_iter=400, reg_covar=1e-6).fit(data)
    assert g.covariances_ is not None
    assert g.covariances_[0][0][1] > g.covariances_[0][1][2]
    assert g.covariances_[0][0][1] > g.covariances_[0][0][2]


def test_2components():
    """
    Simple test with 2 components from clearly delimited data.
    """
    data = np.array(
        [
            [5.1, 10.2, np.nan],
            [4.9, np.nan, np.nan],
            [5.0, 9.9, 1.0],
            [np.nan, 10.1, 0.9],
            [5.2, 10.5, 1.1],
            [np.nan, 11.0, 1.2],
            [7.0, 56.0, 10.0],
            [7.4, 57.0, 11.0],
            [7.1, 54.0, 10.0],
            [7.3, 55.0, 13.0],
        ]
    )
    g = GMMMissing(n_components=2, max_iter=400, reg_covar=1e-6).fit(data)
    assert g.mu_ is not None

    # Sort components by first feature mean to avoid label-switching mismatches
    order = np.argsort(g.mu_[:, 0])
    mu_ = g.mu_[order]

    np.testing.assert_allclose(mu_[0], [5.1, 10.2, 1.0], rtol=0.05)
    np.testing.assert_allclose(mu_[1], [7.2, 55.5, 11.0], rtol=0.05)


def test_compare_with_sklearn_gaussian_mixture_2D():
    """
    Compare GMMMissing parameters against sklearn's GaussianMixture.
    """
    rng = np.random.default_rng(123)

    n_samples_per_cluster = 300
    mean1, cov1 = [0.0, 0.0], [[1.0, 0.3], [0.3, 1.0]]
    mean2, cov2 = [10.0, 10.0], [[2.0, -0.5], [-0.5, 2.0]]

    X1 = rng.multivariate_normal(mean1, cov1, size=n_samples_per_cluster)
    X2 = rng.multivariate_normal(mean2, cov2, size=n_samples_per_cluster)
    X = np.vstack([X1, X2])

    g_sk = GaussianMixture(n_components=2, random_state=42, reg_covar=1e-6).fit(X)
    g_missing = GMMMissing(
        n_components=2, random_state=42, max_iter=400, reg_covar=1e-6
    ).fit(X)

    assert g_missing.mu_ is not None
    assert g_missing.covariances_ is not None

    # Sort components by first feature mean to avoid label-switching mismatches
    sk_order = np.argsort(g_sk.means_[:, 0])
    sk_means = g_sk.means_[sk_order]
    sk_covs = g_sk.covariances_[sk_order]

    missing_order = np.argsort(g_missing.mu_[:, 0])
    missing_means = g_missing.mu_[missing_order]
    missing_covs = g_missing.covariances_[missing_order]

    # Compare fitted means and covariances
    np.testing.assert_allclose(missing_means, sk_means, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(missing_covs, sk_covs, rtol=5e-2, atol=5e-2)


def test_compare_with_sklearn_gaussian_mixture_2D_addnoise():
    """
    Compare GMMMissing parameters against sklearn's GaussianMixture.
    Adds minimal NaN noise to the data for GMMMissing.
    """
    rng = np.random.default_rng(123)

    n_samples_per_cluster = 300
    mean1, cov1 = [0.0, 0.0], [[1.0, 0.3], [0.3, 1.0]]
    mean2, cov2 = [10.0, 10.0], [[2.0, -0.5], [-0.5, 2.0]]

    X1 = rng.multivariate_normal(mean1, cov1, size=n_samples_per_cluster)
    X2 = rng.multivariate_normal(mean2, cov2, size=n_samples_per_cluster)
    X = np.vstack([X1, X2])

    p = 0.01
    missing_matrix = rng.choice([np.nan, 1.0], size=X.shape, p=[p, 1 - p])
    X_noisy = X * missing_matrix

    g_sk = GaussianMixture(n_components=2, random_state=42, reg_covar=1e-6).fit(X)
    g_missing = GMMMissing(
        n_components=2, random_state=42, max_iter=400, reg_covar=1e-6
    ).fit(X_noisy)

    assert g_missing.mu_ is not None
    assert g_missing.covariances_ is not None

    # Sort components by first feature mean to avoid label-switching mismatches
    sk_order = np.argsort(g_sk.means_[:, 0])
    sk_means = g_sk.means_[sk_order]
    sk_covs = g_sk.covariances_[sk_order]

    missing_order = np.argsort(g_missing.mu_[:, 0])
    missing_means = g_missing.mu_[missing_order]
    missing_covs = g_missing.covariances_[missing_order]

    # Compare fitted means and covariances
    np.testing.assert_allclose(missing_means, sk_means, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(missing_covs, sk_covs, rtol=5e-2, atol=5e-2)


def test_compare_with_sklearn_gaussian_mixture_5D_addnoise():
    """
    Compare GMMMissing parameters against sklearn's GaussianMixture in 5D.
    Adds minimal NaN noise to the data for GMMMissing.
    """
    rng = np.random.default_rng(123)

    n_samples_per_cluster = 500

    # Define 5D means and positive-definite covariance matrices
    mean1 = [0.0, 1.0, -1.0, 2.0, 0.5]
    cov1 = [
        [1.5, 0.3, 0.1, 0.0, 0.2],
        [0.3, 1.2, 0.2, 0.1, 0.0],
        [0.1, 0.2, 1.5, -0.3, 0.1],
        [0.0, 0.1, -0.3, 1.0, 0.1],
        [0.2, 0.0, 0.1, 0.1, 1.3],
    ]

    mean2 = [10.0, 8.0, 5.0, -3.0, 7.0]
    cov2 = [
        [2.0, -0.4, 0.2, 0.1, -0.2],
        [-0.4, 1.8, 0.0, 0.3, 0.1],
        [0.2, 0.0, 2.2, -0.5, 0.0],
        [0.1, 0.3, -0.5, 1.5, 0.2],
        [-0.2, 0.1, 0.0, 0.2, 1.9],
    ]

    X1 = rng.multivariate_normal(mean1, cov1, size=n_samples_per_cluster)
    X2 = rng.multivariate_normal(mean2, cov2, size=n_samples_per_cluster)
    X = np.vstack([X1, X2])

    p = 0.1
    missing_matrix = rng.choice([np.nan, 1.0], size=X.shape, p=[p, 1 - p])
    X_noisy = X * missing_matrix

    g_sk = GaussianMixture(n_components=2, random_state=42, reg_covar=1e-6).fit(X)
    g_missing = GMMMissing(
        n_components=2, random_state=42, max_iter=400, reg_covar=1e-6
    ).fit(X_noisy)

    assert g_missing.mu_ is not None
    assert g_missing.covariances_ is not None

    # Sort components by first feature mean to avoid label-switching mismatches
    sk_order = np.argsort(g_sk.means_[:, 0])
    sk_means = g_sk.means_[sk_order]
    sk_covs = g_sk.covariances_[sk_order]

    missing_order = np.argsort(g_missing.mu_[:, 0])
    missing_means = g_missing.mu_[missing_order]
    missing_covs = g_missing.covariances_[missing_order]

    # Compare fitted means and covariances
    np.testing.assert_allclose(missing_means, sk_means, rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(missing_covs, sk_covs, rtol=1e-1, atol=1e-1)


def test_compare_with_sklearn_gaussian_mixture_40D_addnoise():
    """
    Compare GMMMissing parameters against sklearn's GaussianMixture in 40D with 3 components.
    Means and positive-definite covariance matrices are generated programmatically.
    """
    rng = np.random.default_rng(123)
    n_dim = 40
    n_components = 3
    n_samples_per_cluster = (
        2500  # Higher sample size to accommodate 40x40 covariance estimation
    )

    # Programmatically generate means separated along the first dimension for deterministic sorting
    means = [
        rng.normal(loc=i * 20.0, scale=1.0, size=n_dim) for i in range(n_components)
    ]

    # Programmatically generate positive-definite 40x40 covariance matrices (A @ A.T / dim + I)
    covariances = []
    for _ in range(n_components):
        A = rng.normal(size=(n_dim, n_dim))
        cov = (A @ A.T) / n_dim + np.eye(n_dim)
        covariances.append(cov)

    # Generate synthetic clusters
    clusters = [
        rng.multivariate_normal(means[i], covariances[i], size=n_samples_per_cluster)
        for i in range(n_components)
    ]
    X = np.vstack(clusters)

    # Add 10% missingness (NaN noise)
    p = 0.1
    missing_matrix = rng.choice([np.nan, 1.0], size=X.shape, p=[p, 1 - p])
    X_noisy = X * missing_matrix

    # Fit models
    g_sk = GaussianMixture(
        n_components=n_components, random_state=42, reg_covar=1e-5
    ).fit(X)
    g_missing = GMMMissing(
        n_components=n_components, random_state=42, max_iter=400, reg_covar=1e-5
    ).fit(X_noisy)

    assert g_missing.mu_ is not None
    assert g_missing.covariances_ is not None

    # Sort components by first feature mean to prevent label switching
    sk_order = np.argsort(g_sk.means_[:, 0])
    sk_means = g_sk.means_[sk_order]
    sk_covs = g_sk.covariances_[sk_order]

    missing_order = np.argsort(g_missing.mu_[:, 0])
    missing_means = g_missing.mu_[missing_order]
    missing_covs = g_missing.covariances_[missing_order]

    # Compare fitted means and covariances
    np.testing.assert_allclose(missing_means, sk_means, rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(missing_covs, sk_covs, rtol=1e-1, atol=1e-1)
