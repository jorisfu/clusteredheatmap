# pyright: reportExplicitAny=false
from typing import Any
import numpy as np
import numpy.typing as npt
import typing
from typing import Literal

import numpy_typing_compat
import scipy
from scipy.spatial.distance import squareform

from clusteredheatmap.algos.gmm_missing.gmm import GMMMissing
from clusteredheatmap.algos.misc import ecmnmle
from clusteredheatmap.types import DistFun, Vector, PDistFun

import logging

logger = logging.getLogger(__name__)

ScipySupportedDist = Literal[
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]

ChmSupportedDist = Literal[
    "dixon_pds_euclidean",
    "dixon_pds_sqeuclidean",
    "mesquita_eed",
    "eirola_esd_mvn",
    "eirola_esd_gmm",
]

DistFunName = ScipySupportedDist | ChmSupportedDist


class DistanceError(Exception):
    """
    Generic exception used for issues concerning distance calculation
    """

    pass


def _dixon_pds_euclidean(a: Vector, b: Vector, sqrt: bool = True) -> np.float64:
    """
    Partial Distance Strategy as proposed by Dixon.
    See "Pattern Recognition with Partly Missing Data" by John K. Dixon.
    """
    nan_mask = np.isnan(a) | np.isnan(b)

    if len(a) == np.sum(nan_mask):
        raise DistanceError(
            "The two input vectors have no overlapping observed features, cannot estimate distance using PDS."
        )

    weight = len(nan_mask) / (len(nan_mask) - sum(nan_mask))

    masked_a = a[~nan_mask]
    masked_b = b[~nan_mask]

    d = scipy.spatial.distance.sqeuclidean(masked_a, masked_b)
    res = weight * d

    if sqrt:
        res = np.sqrt(res)

    return np.float64(res)


def dixon_pds_euclidean(a: Vector, b: Vector) -> np.float64:
    return _dixon_pds_euclidean(a, b, sqrt=True)


def dixon_pds_sqeuclidean(a: Vector, b: Vector) -> np.float64:
    return _dixon_pds_euclidean(a, b, sqrt=False)


def cc_pearson(a: Vector, b: Vector) -> np.float64:
    nan_mask = np.isnan(a) | np.isnan(b)

    masked_a = a[~nan_mask]
    masked_b = b[~nan_mask]
    d = scipy.spatial.distance.correlation(masked_a, masked_b)

    return d


def mesquita_eed(
    data: npt.NDArray[np.float64], min_k: int = 1, max_k: int = 4, max_iter: int = 200
) -> npt.NDArray[np.float64]:
    """
    Expected Euclidean Distance as proposed by Mesquita et al. See http://dx.doi.org/10.1016/j.neucom.2016.12.081.
    Implemented as described in Algorithm 1.

    Assumes distances are Nakagami-distributed. Data distribution modeled via a Gaussian mixture distribution.
    """
    n_observations, n_features = data.shape

    # TODO: Section 5.3, Implement Bayesian Information criterion to get best model
    k = 2
    used_model = GMMMissing(k, max_iter=max_iter).fit(data)
    n_components = used_model.n_components
    estimated_covars = used_model.covariances_
    estimated_means = used_model.mu_
    estimated_resp = used_model.predict_proba()

    pdist = squareform(np.full((n_observations, n_observations), 0.0))

    # Conditional mean for each component and each vector (lines 7, 9)
    cond_mu = [[None for _i in range(n_observations)] for _j in range(n_components)]
    np.full((n_components, n_observations, n_features), np.nan)
    # Conditional covar for each component and each vector (lines 8, 10)
    cond_Sigma = np.full((n_components, n_observations, n_features, n_features), np.nan)

    for i in range(n_observations):
        for j in range(i + 1, n_observations):

            # padded means and covars (lines 13-20)
            pad_mu = np.full((n_components, n_features), 0.0)
            pad_Sigma = np.full((n_components, n_features, n_features), 0.0)

            ## Initialization
            mis_i = np.isnan(data[i])
            obs_i = ~mis_i
            mis_j = np.isnan(data[j])
            obs_j = ~mis_j
            eta_hat = 0.0

            ## Condition each of the GMM components on the observed values of both Xi and Xj.
            # NOTE that this is technically a small optimization compared to the 
            # pseudocode given, but it's quite the trivial one (just memoization)
            for c in range(n_components):
                mu_c = estimated_means[c]
                Sigma_c = estimated_covars[c]
            
                if np.isnan(cond_mu[c][i][0]):
                    Sigma_c_oo_i = Sigma_c[np.ix_(obs_i, obs_i)]
                    Sigma_c_mo_i = Sigma_c[np.ix_(mis_i, obs_i)]
                    Sigma_c_om_i = Sigma_c[np.ix_(obs_i, mis_i)]
                    Sigma_c_mm_i = Sigma_c[np.ix_(mis_i, mis_i)]

                    beta_i = Sigma_c_mo_i @ np.linalg.inv(Sigma_c_oo_i)

                    cond_mu[c][i] = mu_c[mis_i] + beta_i @ (data[i][obs_i] - mu_c[obs_i])
                    cond_Sigma[c][i] = Sigma_c_mm_i - beta_i @ Sigma_c_om_i

                if np.isnan(cond_mu[c][j][0]):
                    Sigma_c_oo_j = Sigma_c[np.ix_(obs_j, obs_j)]
                    Sigma_c_mo_j = Sigma_c[np.ix_(mis_j, obs_j)]
                    Sigma_c_om_j = Sigma_c[np.ix_(obs_j, mis_j)]
                    Sigma_c_mm_j = Sigma_c[np.ix_(mis_j, mis_j)]

                    beta_j = Sigma_c_mo_j @ np.linalg.inv(Sigma_c_oo_j)

                    cond_mu[c][j] = mu_c[mis_j] + beta_j @ (data[j][obs_j] - mu_c[obs_j])
                    cond_Sigma[c][j] = Sigma_c_mm_j - beta_j @ Sigma_c_om_j

                ## Compute padded conditional mean vectors and conditional covariance matrices 
                ## of Xi − Xj for each GMM component.
                # pad_mu_c = np.full((n_features), 0.0)
                pad_mu[c][obs_i] += data[i][obs_i]
                pad_mu[c][obs_j] += data[j][obs_j]
                pad_mu[c][mis_i] += cond_mu[c][i]
                pad_mu[c][mis_j] += cond_mu[c][j]

                # NOTE: There is likely a typo in the original paper in lines 19,20
                # since adding pad_Sigma_c again would always result in an empty matrix
                # pad_Sigma[c] = np.full((n_features, n_features), 0.0)
                pad_Sigma[c][np.ix_(mis_i, mis_i)] += cond_Sigma[c][i]
                pad_Sigma[c][np.ix_(mis_j, mis_j)] += cond_Sigma[c][j]

            ## Compute eta_hat
            for d in range(n_features):
                m = 0.0
                s = 0.0

                for c in range(n_components):
                    m += used_model.alpha_[c] * pad_mu[c][d]
                    s += used_model.alpha_[c] * (pad_mu[c][d] ** 2 + pad_Sigma[c][d][d])

                v = s - m * m
                eta_hat += 4 * m * m * v + 2 * v * v

            pdist[n_observations * i + j - ((i + 2) * (i + 1)) // 2] = eta_hat

    return pdist


def eirola_esd_gmm(
    data: npt.NDArray[np.float64], min_k: int = 1, max_k: int = 4, max_iter: int = 200
) -> npt.NDArray[np.float64]:
    """
    Expected Squared Distance as proposed by Eirola et al. See http://dx.doi.org/10.1016/j.neucom.2013.07.050
    Using Mixture of Gaussians for estimation.
    The GMM that minimises the corrected AIC gets selected (from all GMMs with k in [min_k, max_k]).
    Algorithm implemented as described in section 3.

    Additional parameters:
    :param min_k: Minimum number of Gaussian components to try for GMM.
    :param max_k: Maximum number of Gaussian components to try for GMM.
    :param max_iter: Maximum allowed iterations for each GMM fitting.
    """
    n_observations, n_features = data.shape

    def corrected_aic(log_likelihood: np.floating | float, k: int) -> np.float64:
        """
        See section 2.6. Computes the corrected Akaike information
        criterion.

        :param log_likelihood: log_likelihood of the model on the data
        :param k: number of Gaussians used in the model
        """
        P = (
            k * n_features + (k - 1) + (0.5 * k * n_features * (n_features + 1))
        )  # Equation (9), no. of free parameters
        return (
            -2 * log_likelihood + 2 * P + (2 * P * (P + 1)) / (n_observations - P - 1)
        )

    ##
    ## Steps 1-2: Fit models and get AICc and LL for each model
    ##

    models: list[GMMMissing] = []
    n_models = max_k - min_k + 1
    log_likelihood = np.full((n_models), 0.0)
    aicc = np.full((n_models), 0.0)

    for k in range(min_k, max_k + 1):
        model = GMMMissing(k, max_iter=max_iter).fit(data)
        models.append(model)

        ll = model.lower_bound_
        log_likelihood[k - min_k] = ll
        aicc[k - min_k] = corrected_aic(ll, k)

    ##
    ## Step 3: Select model with minimal AICc and get conditional means/covars
    ##

    used_model = models[np.argmin(aicc)]
    logging.info("Using GMM with " + str(used_model.n_components) + " components")

    s = np.full(
        (n_observations), 0.0
    )  # Holds the summed variance terms for each observation (equation (11))
    imputed_data = np.full(data.shape, 0.0)
    estimated_covars = used_model.covariances_
    estimated_means = used_model.mu_
    estimated_resp = used_model.predict_proba()

    assert estimated_means is not None
    assert estimated_covars is not None

    missing = np.isnan(data)
    observed = ~missing

    for i in range(n_observations):
        obs_i = observed[i]
        mis_i = missing[i]

        imp_x_i = np.full((n_features), 0.0)
        imp_Sigma_i = np.full((n_features, n_features), 0.0)

        for k in range(used_model.n_components):
            mu_k = estimated_means[k]
            t_ik = estimated_resp[i][k]
            Sigma_k_oo_i = estimated_covars[k][np.ix_(obs_i, obs_i)]
            Sigma_k_mo_i = estimated_covars[k][np.ix_(mis_i, obs_i)]
            Sigma_k_om_i = estimated_covars[k][np.ix_(obs_i, mis_i)]
            Sigma_k_mm_i = estimated_covars[k][np.ix_(mis_i, mis_i)]

            # See equations (6) and (7)
            beta = Sigma_k_mo_i @ np.linalg.inv(Sigma_k_oo_i)

            imp_mu_ik_mis = mu_k[mis_i] + beta @ (data[i][obs_i] - mu_k[obs_i])
            imp_x_ik = data[i].copy()
            imp_x_ik[mis_i] = imp_mu_ik_mis

            imp_Sigma_ik_mm = Sigma_k_mm_i - beta @ Sigma_k_om_i
            imp_Sigma_ik = np.full((n_features, n_features), 0.0)
            imp_Sigma_ik[np.ix_(mis_i, mis_i)] = imp_Sigma_ik_mm

            # See equation (12)
            imp_x_i += t_ik * imp_x_ik
            imputed_data[i] = imp_x_i

            imp_Sigma_i += t_ik * (imp_Sigma_ik + imp_x_ik @ imp_x_ik.transpose())

        # TODO: Check these two variants
        # imp_Sigma_i -= used_model.n_components * (imp_x_i @ imp_x_i.transpose())
        imp_Sigma_i -= imp_x_i @ imp_x_i.transpose()

        s[i] = np.linalg.trace(imp_Sigma_i)

    ##
    ## Step 4: Esimate distances
    ##

    # print("Data")
    # print(data)
    # print("imputed Data")
    # print(imputed_data)

    pdist = scipy.spatial.distance.pdist(imputed_data, "sqeuclidean")
    oldpdist = pdist.copy()

    for i in range(n_observations):
        for j in range(i + 1, n_observations):
            pdist[n_observations * i + j - ((i + 2) * (i + 1)) // 2] += (
                s[i] + s[j]
            )  # Apply correction

    # print("Old pdist")
    # print(pdist)
    # print("Difference to corrected pdist")
    # print(pdist - oldpdist)
    # print("Corrected pdist")
    # print(pdist)

    return pdist


def eirola_esd_mvn(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Expected Squared Distance as proposed by Eirola et al. See [TODO].
    Algorithm implemented as described in section 3.4
    """

    n_observations, n_features = data.shape

    mean, cov = ecmnmle(data, max_iterations=70)
    s_squared = np.full((n_observations), 0.0)
    imputed_data = data.copy()

    for i in range(n_observations):
        x_i = data[i].copy()
        missing = np.isnan(x_i)
        observed = ~missing

        if not np.any(missing):
            continue

        # Following calculation is basically the same as in the E step of the ecmnmle algorithm.
        # Using the conditional means and variances as also outlined in the ESD paper
        # (section 3.2).
        # Not employing any additional safeguards here as the converged result
        # _should_ work fine.
        x_i_o = x_i[observed]
        mu_o = mean[observed]
        mu_m = mean[missing]

        cov_oo = cov[np.ix_(observed, observed)]
        cov_mo = cov[np.ix_(missing, observed)]
        cov_om = cov[np.ix_(observed, missing)]
        cov_mm = cov[np.ix_(missing, missing)]

        inv_cov_oo = np.linalg.pinv(cov_oo, hermitian=True)
        beta = cov_mo @ inv_cov_oo

        conditional_mean = mu_m + beta @ (x_i_o - mu_o)
        conditional_cov = cov_mm - beta @ cov_om

        diag_sum = np.linalg.trace(conditional_cov)
        s_squared[i] = diag_sum

        x_i[missing] = conditional_mean
        imputed_data[i] = x_i

    pdist = scipy.spatial.distance.pdist(imputed_data, "sqeuclidean")

    for i in range(n_observations):
        for j in range(i + 1, n_observations):
            pdist[n_observations * i + j - ((i + 2) * (i + 1)) // 2] += (
                s_squared[i] + s_squared[j]
            )

    return pdist


_mapping: dict[DistFunName, DistFun] = {
    "dixon_pds_euclidean": dixon_pds_euclidean,
    "dixon_pds_sqeuclidean": dixon_pds_sqeuclidean,
}

_pdist_mapping: dict[DistFunName, PDistFun] = {
    "eirola_esd_mvn": eirola_esd_mvn,
    "eirola_esd_gmm": eirola_esd_gmm,
}


def get_preferred_pdist_implementation(
    distance: DistFunName | DistFun,
    distance_args: dict[str, Any] | None,
) -> PDistFun:
    """
    From a given name of a distance function to use or a distance function, returns
    a callable function compatible with pdist to create
    a condensed distance matrix from an array of observation
    vectors.
    """

    if distance_args is None:
        distance_args = {}

    # Passed distance function
    if callable(distance):
        return lambda mat: scipy.spatial.distance.pdist(mat, distance)

    # Scipy supported distance function by name
    if distance in list(typing.get_args(ScipySupportedDist)):
        return lambda mat: scipy.spatial.distance.pdist(mat, distance, **distance_args)

    # CHM supported distance function by name
    if distance in _mapping.keys():
        fun_with_args = lambda a, b: _mapping[distance](a, b, **distance_args)
        return lambda mat: scipy.spatial.distance.pdist(mat, fun_with_args)

    # CHM supported pdist-compatible distance function by name
    if distance in _pdist_mapping.keys():
        return lambda mat: _pdist_mapping[distance](mat, **distance_args)

    raise ValueError(f"Distance function {distance} not supported")
