"""
Gaussian Mixture Model for Incomplete Data.

This module implements the GMM algorithm with missing data handling
as described in "Gaussian Mixture Model Clustering with Incomplete Data" by Zhang et al.

This file was taken from the `gmm-imputation-clustering` library as found in
https://github.com/khang3004/gmm-imputation-clustering/blob/main/src/gmm_missing/core/gmm.py
(last accessed 2026-07-27)
and is licensed under the MIT license.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class GMMMissing(BaseEstimator, ClusterMixin):
    """
    Gaussian Mixture Model Clustering with Incomplete Data.

    This class implements an EM-style algorithm that simultaneously:
    1. Clusters data using GMM
    2. Imputes missing values during the M-step

    The algorithm alternates between:
    - E-step: Compute responsibilities given current parameters
    - M-step (params): Update GMM parameters (means, covariances, weights)
    - M-step (data): Impute missing values using conditional expectations

    Parameters
    ----------
    n_components : int, default=3
        Number of mixture components (clusters).
    max_iter : int, default=100
        Maximum number of EM iterations.
    tol : float, default=1e-4
        Convergence tolerance for log-likelihood change.
    reg_covar : float, default=1e-6
        Regularization term added to covariance diagonals for numerical stability.
    random_state : int | None, default=None
        Random seed for reproducibility.
    init_params : {'kmeans', 'random'}, default='kmeans'
        Method for initializing GMM parameters.

    Attributes
    ----------
    mu_ : ndarray of shape (n_components, n_features)
        Component means.
    covariances_ : ndarray of shape (n_components, n_features, n_features)
        Component covariance matrices.
    alpha_ : ndarray of shape (n_components,)
        Mixing weights (component probabilities).
    mask_ : ndarray of shape (n_samples, n_features)
        Boolean mask indicating missing values (True = missing).
    X_final_ : ndarray of shape (n_samples, n_features)
        Final imputed dataset.
    n_iter_ : int
        Actual number of iterations performed.
    lower_bound_ : float
        Final log-likelihood value.

    Examples
    --------
    >>> import numpy as np
    >>> from gmm_missing.core import GMMMissing
    >>> X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, np.nan]])
    >>> model = GMMMissing(n_components=2, random_state=42)
    >>> model.fit(X)
    >>> labels = model.predict()
    >>> X_imputed = model.X_final_
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 100,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: Optional[int] = None,
        init_params: Literal["kmeans", "random"] = "kmeans",
    ) -> None:
        """Initialize GMM Missing estimator."""
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")
        if reg_covar < 0:
            raise ValueError(f"reg_covar must be >= 0, got {reg_covar}")
        if init_params not in ("kmeans", "random"):
            raise ValueError(
                f"init_params must be 'kmeans' or 'random', got {init_params}"
            )

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.init_params = init_params

        # Initialize attributes that will be set during fit
        self.mu_: Optional[NDArray[np.floating]] = None
        self.covariances_: Optional[NDArray[np.floating]] = None
        self.alpha_: Optional[NDArray[np.floating]] = None
        self.mask_: Optional[NDArray[np.bool_]] = None
        self.X_final_: Optional[NDArray[np.floating]] = None
        self.n_iter_: int = 0
        self.lower_bound_: float = -np.inf

    def _initialize_parameters(self, X: NDArray[np.floating]) -> None:
        """
        Initialize GMM parameters using K-means or random selection.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data (should have initial imputation already applied).
        """
        n_samples, n_features = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)
            rng = np.random.RandomState(self.random_state)
        else:
            rng = np.random.RandomState()

        if self.init_params == "kmeans":
            logger.debug("Initializing parameters using K-means")
            kmeans = KMeans(
                n_clusters=self.n_components,
                random_state=self.random_state,
                n_init="auto",
            )
            labels = kmeans.fit_predict(X)
            self.mu_ = kmeans.cluster_centers_.copy()
        else:
            logger.debug("Initializing parameters randomly")
            random_indices = rng.choice(
                n_samples, size=self.n_components, replace=False
            )
            self.mu_ = X[random_indices].copy()
            labels = rng.randint(0, self.n_components, size=n_samples)

        # Initialize mixing weights uniformly
        self.alpha_ = np.ones(self.n_components, dtype=np.float64) / self.n_components

        # Initialize covariances as identity matrices
        self.covariances_ = np.array(
            [np.eye(n_features, dtype=np.float64) for _ in range(self.n_components)]
        )

        # Estimate initial parameters based on cluster assignments
        for k in range(self.n_components):
            mask_k = labels == k
            n_k = np.sum(mask_k)

            if n_k > 0:
                self.alpha_[k] = n_k / n_samples
                X_k = X[mask_k]

                # Compute sample covariance
                if n_k > 1:
                    cov_k = np.cov(X_k, rowvar=False)
                    # Ensure 2D array for single feature case
                    if cov_k.ndim == 0:
                        cov_k = np.array([[cov_k]])
                else:
                    cov_k = np.eye(n_features)

                # Add regularization for numerical stability
                cov_k += np.eye(n_features) * self.reg_covar
                self.covariances_[k] = cov_k

        logger.info(
            f"Initialized GMM parameters: {self.n_components} components, "
            f"{n_features} features"
        )

    def _e_step(self, X: NDArray[np.floating]) -> Tuple[float, NDArray[np.floating]]:
        """
        Expectation step: compute responsibilities and log-likelihood.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Current (imputed) data.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of the data under current parameters.
        gamma : ndarray of shape (n_samples, n_components)
            Responsibility matrix (posterior probabilities).
        """
        n_samples = X.shape[0]
        log_resp = np.zeros((n_samples, self.n_components), dtype=np.float64)

        for k in range(self.n_components):
            try:
                # Create multivariate normal distribution for component k
                rv = multivariate_normal(
                    mean=self.mu_[k], cov=self.covariances_[k], allow_singular=True
                )
                log_prob = rv.logpdf(X)
                log_resp[:, k] = np.log(self.alpha_[k] + 1e-300) + log_prob

            except np.linalg.LinAlgError as e:
                # Fallback: add extra regularization if covariance is singular
                logger.warning(
                    f"Singular covariance detected for component {k}, "
                    f"adding extra regularization"
                )
                pseudo_cov = self.covariances_[k] + np.eye(X.shape[1]) * (
                    self.reg_covar * 10
                )
                rv = multivariate_normal(
                    mean=self.mu_[k], cov=pseudo_cov, allow_singular=True
                )
                log_prob = rv.logpdf(X)
                log_resp[:, k] = np.log(self.alpha_[k] + 1e-300) + log_prob

        # Normalize responsibilities using log-sum-exp trick for numerical stability
        log_prob_norm = logsumexp(log_resp, axis=1)
        log_resp_normalized = log_resp - log_prob_norm[:, np.newaxis]
        gamma = np.exp(log_resp_normalized)

        # Compute total log-likelihood
        log_likelihood = log_prob_norm.sum()

        return log_likelihood, gamma

    def _m_step_params(
        self, X: NDArray[np.floating], gamma: NDArray[np.floating]
    ) -> None:
        """
        Maximization step: update GMM parameters (alpha, mu, Sigma).

        Implements equations 9, 10, 11 from Zhang et al.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Current (imputed) data.
        gamma : ndarray of shape (n_samples, n_components)
            Responsibility matrix.
        """
        n_samples, n_features = X.shape
        N_k = gamma.sum(axis=0)  # Effective number of points per component

        # Update mixing weights (Equation 9)
        self.alpha_ = N_k / n_samples

        for k in range(self.n_components):
            if N_k[k] < 1e-15:
                logger.warning(
                    f"Component {k} has negligible responsibility, skipping update"
                )
                continue

            # Update mean (Equation 10)
            self.mu_[k] = (gamma[:, k : k + 1] * X).sum(axis=0) / N_k[k]

            # Update covariance (Equation 11)
            diff = X - self.mu_[k]
            weighted_outer_product = np.einsum(
                "n,nij->ij", gamma[:, k], np.einsum("ni,nj->nij", diff, diff)
            )

            self.covariances_[k] = weighted_outer_product / N_k[k]
            self.covariances_[k] += np.eye(n_features) * self.reg_covar

        logger.debug(
            f"M-step completed: updated parameters for {self.n_components} components"
        )

    def _m_step_data(
        self,
        X: NDArray[np.floating],
        mask: NDArray[np.bool_],
        gamma: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Maximization step: impute missing data using conditional expectations.

        Implements Equation 14 from Zhang et al.
        For each sample with missing values, computes the optimal imputation
        as a weighted combination of conditional expectations from each component.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Current data with some values already imputed.
        mask : ndarray of shape (n_samples, n_features)
            Boolean mask where True indicates missing values.
        gamma : ndarray of shape (n_samples, n_components)
            Responsibility matrix.

        Returns
        -------
        X_updated : ndarray of shape (n_samples, n_features)
            Data with improved imputations for missing values.
        """
        n_samples, n_features = X.shape
        X_updated = X.copy()

        # Precompute precision matrices (inverse covariances) for efficiency
        precisions = np.zeros_like(self.covariances_)
        for k in range(self.n_components):
            try:
                precisions[k] = np.linalg.inv(self.covariances_[k])
            except np.linalg.LinAlgError:
                # Add regularization if inversion fails
                reg_cov = self.covariances_[k] + np.eye(n_features) * (
                    self.reg_covar * 10
                )
                precisions[k] = np.linalg.inv(reg_cov)
                logger.warning(
                    f"Added regularization for precision matrix of component {k}"
                )

        # Impute missing values for each sample
        for i in range(n_samples):
            m_part = mask[i]  # Boolean mask for missing indices

            # Skip if no missing values
            if not np.any(m_part):
                continue

            o_part = ~m_part  # Observed indices
            dim_m = np.sum(m_part)
            x_i_o = X[i, o_part]  # Observed values

            # Accumulate weighted precision matrices and terms
            sum_L_mm = np.zeros((dim_m, dim_m), dtype=np.float64)
            sum_term2 = np.zeros(dim_m, dtype=np.float64)

            for k in range(self.n_components):
                gamma_ik = gamma[i, k]

                # Skip components with negligible responsibility
                if gamma_ik < 1e-15:
                    continue

                Lambda_k = precisions[k]

                # Extract relevant submatrices of precision matrix
                L_mm = Lambda_k[np.ix_(m_part, m_part)]  # Σ_mm^{-1}
                L_mo = Lambda_k[np.ix_(m_part, o_part)]  # Σ_mo^{-1}

                mu_m = self.mu_[k, m_part]
                mu_o = self.mu_[k, o_part]

                # Accumulate weighted terms
                sum_L_mm += gamma_ik * L_mm
                term2 = L_mm @ mu_m - L_mo @ (x_i_o - mu_o)
                sum_term2 += gamma_ik * term2

            # Solve for optimal imputation
            try:
                x_i_m_updated = np.linalg.solve(sum_L_mm, sum_term2)
                X_updated[i, m_part] = x_i_m_updated
            except np.linalg.LinAlgError:
                logger.warning(
                    f"Could not solve linear system for sample {i}, "
                    f"keeping original imputation"
                )
                # Keep original value as fallback

        return X_updated

    def fit(self, X_incomplete: NDArray[np.floating]) -> "GMMMissing":
        """
        Fit GMM model with incomplete data using alternating EM optimization.

        The algorithm alternates between:
        1. E-step: Compute responsibilities
        2. M-step (params): Update GMM parameters
        3. M-step (data): Impute missing values

        Parameters
        ----------
        X_incomplete : ndarray of shape (n_samples, n_features)
            Input data with missing values represented as NaN.

        Returns
        -------
        self : GMMMissing
            Fitted estimator.

        Raises
        ------
        ValueError
            If input contains invalid values or all values are missing.
        """
        X = np.asarray(X_incomplete, dtype=np.float64)

        # Validate input
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature")

        # Create missing value mask
        self.mask_ = np.isnan(X)

        # Check for completely missing rows or columns
        all_missing_rows = np.all(self.mask_, axis=1)
        if np.any(all_missing_rows):
            raise ValueError(
                f"Samples {np.where(all_missing_rows)[0]} have all missing values. "
                "Each sample must have at least one observed value."
            )

        # Initial imputation using column means
        logger.info("Starting initial imputation with column means")
        col_means = np.nanmean(X, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)  # Handle all-NaN columns

        # Fill missing values with column means
        X_imputed = X.copy()
        missing_indices = np.where(self.mask_)
        X_imputed[missing_indices] = np.take(col_means, missing_indices[1])

        # Initialize GMM parameters
        self._initialize_parameters(X_imputed)
        self.lower_bound_ = -np.inf

        # Run EM iterations
        logger.info(
            f"Starting EM algorithm with max_iter={self.max_iter}, tol={self.tol}"
        )

        for iteration in range(self.max_iter):
            prev_lower_bound = self.lower_bound_

            # E-step: compute responsibilities
            log_likelihood, gamma = self._e_step(X_imputed)
            self.lower_bound_ = log_likelihood

            # Check convergence
            change = abs(self.lower_bound_ - prev_lower_bound)
            if change < self.tol:
                logger.info(f"Converged at iteration {iteration} (change={change:.6f})")
                break

            # M-step: update parameters
            self._m_step_params(X_imputed, gamma)

            # M-step: impute missing data
            X_imputed = self._m_step_data(X_imputed, self.mask_, gamma)

            # Log progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                logger.info(
                    f"Iteration {iteration + 1}: log-likelihood={self.lower_bound_:.4f}, "
                    f"change={change:.6f}"
                )
        else:
            logger.warning(f"Did not converge within {self.max_iter} iterations")

        # Store final results
        self.X_final_ = X_imputed
        self.n_iter_ = iteration + 1

        logger.info(
            f"Fitting completed: {self.n_iter_} iterations, "
            f"final log-likelihood={self.lower_bound_:.4f}"
        )

        return self

    def predict(self, X: Optional[NDArray[np.floating]] = None) -> NDArray[np.int_]:
        """
        Predict cluster labels for data.

        Parameters
        ----------
        X : ndarray, optional
            Data to predict labels for. If None, uses the fitted imputed data.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each sample.

        Raises
        ------
        ValueError
            If called before fitting.
        """
        if self.mu_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if X is None:
            X = self.X_final_
        else:
            X = np.asarray(X, dtype=np.float64)
            # Handle missing values in new data by simple mean imputation
            if np.any(np.isnan(X)):
                logger.warning("Input contains NaN values, filling with column means")
                col_means = np.nanmean(X, axis=0)
                col_means = np.nan_to_num(col_means, nan=0.0)
                X[np.isnan(X)] = np.take(col_means, np.where(np.isnan(X))[1])

        _, gamma = self._e_step(X)
        return np.argmax(gamma, axis=1)

    def predict_proba(
        self, X: Optional[NDArray[np.floating]] = None
    ) -> NDArray[np.floating]:
        """
        Predict posterior probabilities (responsibilities) for data.

        Parameters
        ----------
        X : ndarray, optional
            Data to predict probabilities for. If None, uses the fitted imputed data.

        Returns
        -------
        gamma : ndarray of shape (n_samples, n_components)
            Posterior probability of each sample belonging to each component.

        Raises
        ------
        ValueError
            If called before fitting.
        """
        if self.mu_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if X is None:
            X = self.X_final_
        else:
            X = np.asarray(X, dtype=np.float64)
            if np.any(np.isnan(X)):
                logger.warning("Input contains NaN values, filling with column means")
                col_means = np.nanmean(X, axis=0)
                col_means = np.nan_to_num(col_means, nan=0.0)
                X[np.isnan(X)] = np.take(col_means, np.where(np.isnan(X))[1])

        _, gamma = self._e_step(X)
        return gamma

    def score(
        self, X: NDArray[np.floating], y: Optional[NDArray[np.int_]] = None
    ) -> float:
        """
        Compute the log-likelihood of data under the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of X under the model.
        """
        if self.mu_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        if np.any(np.isnan(X)):
            logger.warning("Input contains NaN values, filling with column means")
            col_means = np.nanmean(X, axis=0)
            col_means = np.nan_to_num(col_means, nan=0.0)
            X[np.isnan(X)] = np.take(col_means, np.where(np.isnan(X))[1])

        log_likelihood, _ = self._e_step(X)
        return log_likelihood
