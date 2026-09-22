# pyright: reportExplicitAny=false
from typing import Any
import numpy as np
import numpy.typing as npt
import typing
from typing import Literal

import scipy
from scipy.spatial.distance import squareform
import nandist

from clusteredheatmap.algos.gmm_missing.gmm import GMMMissing
from clusteredheatmap.algos.modelselection import (
    get_best_gmm,
)
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

NandistSupportedDist = Literal[
    "nandist_chebyshev",
    "nandist_cityblock",
    "nandist_cosine",
    "nandist_euclidean",
    "nandist_minkowski",
]

DistFunName = ScipySupportedDist | ChmSupportedDist


class DistanceError(Exception):
    """
    Generic exception used for issues concerning distance calculation
    """

    pass


def dixon_pds_sqeuclidean(a: Vector, b: Vector) -> np.float64:
    """
    Partial Distance Strategy as proposed by Dixon.
    Estimates the squared Euclidean distance.
    See "Pattern Recognition with Partly Missing Data" by John K. Dixon.
    """
    nan_mask = np.isnan(a) | np.isnan(b)

    if len(a) == np.sum(nan_mask):
        raise DistanceError(
            "The two input vectors have no overlapping observed features, cannot estimate distance using PDS."
        )

    weight = len(a) / (len(a) - sum(nan_mask))

    masked_a = a[~nan_mask]
    masked_b = b[~nan_mask]

    d = scipy.spatial.distance.sqeuclidean(masked_a, masked_b)
    res = weight * d

    return np.float64(res)


def dixon_pds_euclidean(a: Vector, b: Vector) -> np.float64:
    """
    Wrapper for dixon_pds_sqeuclidean. Returns the square root
    of the estimated squared Euclidean distance.
    """
    return np.sqrt(dixon_pds_sqeuclidean(a, b))


def mesquita_eed(
    data: npt.NDArray[np.float64],
    min_k: int = 1,
    max_k: int = 4,
    max_iter: int = 200,
    gmm: GMMMissing | None = None,
) -> npt.NDArray[np.float64]:
    """
    Expected Euclidean Distance as proposed by Mesquita et al. See http://dx.doi.org/10.1016/j.neucom.2016.12.081.
    Implemented as described in Algorithm 1.

    Assumes distances are Nakagami-distributed. Data distribution modeled via a Gaussian mixture distribution.

    Additional parameters

    :param min_k: Minimum number of Gaussian components to try for GMM.
    :param max_k: Maximum number of Gaussian components to try for GMM.
    :param max_iter: Maximum allowed iterations for each GMM fitting.
    :param gmm: Pre-computed Gausssian Mixture Model to use instead of attempting multiple models
    """
    n_observations, n_features = data.shape

    used_model = gmm
    if used_model is None:
        used_model = get_best_gmm(min_k, max_k, max_iter, "BIC", data)

    n_components = used_model.n_components
    estimated_covars = used_model.covariances_
    estimated_means = used_model.mu_
    estimated_resp = used_model.predict_proba()

    pdist = squareform(np.full((n_observations, n_observations), 0.0))

    # Conditional mean for each component and each vector (lines 7, 9)
    cond_mu = [[None for _i in range(n_observations)] for _j in range(n_components)]
    # np.full((n_components, n_observations, n_features), np.nan)
    # Conditional covar for each component and each vector (lines 8, 10)
    cond_Sigma = [[None for _i in range(n_observations)] for _j in range(n_components)]
    # np.full((n_components, n_observations, n_features, n_features), np.nan)

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
            var_z = 0.0
            expected_z = 0.0

            ## Added from original: short circuit if everything is known
            if not np.any(mis_i) and not np.any(mis_j):
                pdist[n_observations * i + j - ((i + 2) * (i + 1)) // 2] = (
                    scipy.spatial.distance.euclidean(data[i], data[j])
                )
                continue

            ## Condition each of the GMM components on the observed values of both Xi and Xj.
            # NOTE that this is technically a small optimization compared to the
            # pseudocode given, but it's quite the trivial one (just memoization)
            for c in range(n_components):
                mu_c = estimated_means[c]
                Sigma_c = estimated_covars[c]

                if cond_mu[c][i] is None:
                    Sigma_c_oo_i = Sigma_c[np.ix_(obs_i, obs_i)]
                    Sigma_c_mo_i = Sigma_c[np.ix_(mis_i, obs_i)]
                    Sigma_c_om_i = Sigma_c[np.ix_(obs_i, mis_i)]
                    Sigma_c_mm_i = Sigma_c[np.ix_(mis_i, mis_i)]

                    beta_i = Sigma_c_mo_i @ np.linalg.inv(Sigma_c_oo_i)

                    cond_mu[c][i] = mu_c[mis_i] + beta_i @ (
                        data[i][obs_i] - mu_c[obs_i]
                    )
                    cond_Sigma[c][i] = Sigma_c_mm_i - beta_i @ Sigma_c_om_i

                if cond_mu[c][j] is None:
                    Sigma_c_oo_j = Sigma_c[np.ix_(obs_j, obs_j)]
                    Sigma_c_mo_j = Sigma_c[np.ix_(mis_j, obs_j)]
                    Sigma_c_om_j = Sigma_c[np.ix_(obs_j, mis_j)]
                    Sigma_c_mm_j = Sigma_c[np.ix_(mis_j, mis_j)]

                    beta_j = Sigma_c_mo_j @ np.linalg.inv(Sigma_c_oo_j)

                    cond_mu[c][j] = mu_c[mis_j] + beta_j @ (
                        data[j][obs_j] - mu_c[obs_j]
                    )
                    cond_Sigma[c][j] = Sigma_c_mm_j - beta_j @ Sigma_c_om_j

                ## Compute padded conditional mean vectors and conditional covariance matrices
                ## of Xi − Xj for each GMM component.
                pad_mu[c][obs_i] += data[i][obs_i]
                pad_mu[c][obs_j] -= data[j][obs_j]  # NOTE: Changed this to -=
                pad_mu[c][mis_i] += cond_mu[c][i]
                pad_mu[c][mis_j] -= cond_mu[c][j]  # NOTE: Changed this to -=

                # NOTE: There is likely a typo in the original paper in lines 19,20
                # since adding pad_Sigma_c again would always result in an empty matrix
                # pad_Sigma[c] = np.full((n_features, n_features), 0.0)
                pad_Sigma[c][np.ix_(mis_i, mis_i)] += cond_Sigma[c][i]
                pad_Sigma[c][np.ix_(mis_j, mis_j)] += cond_Sigma[c][j]

            ## Compute eta_hat (Note: this is Var[z] aka variance of the ESD)
            ## We also compute E[z] here
            for d in range(n_features):
                m = 0.0
                s = 0.0

                for c in range(n_components):
                    m += used_model.alpha_[c] * pad_mu[c][d]
                    s += used_model.alpha_[c] * (pad_mu[c][d] ** 2 + pad_Sigma[c][d][d])

                v = s - m * m
                var_z += 4 * m**2 * v + 2 * v**2
                expected_z += m**2 + v

            ## Eq. (6)
            nakagami_m = expected_z**2 / var_z
            nakagami_Omega = expected_z

            ## Eq. (5)
            eed = scipy.stats.nakagami.mean(nakagami_m, scale=np.sqrt(nakagami_Omega))
            if np.isnan(eed):
                print(nakagami_m, nakagami_Omega)
                print(scipy.special.gamma(nakagami_m))

            pdist[n_observations * i + j - ((i + 2) * (i + 1)) // 2] = eed

    return pdist


def eirola_esd_gmm(
    data: npt.NDArray[np.float64],
    min_k: int = 1,
    max_k: int = 4,
    max_iter: int = 200,
    gmm: GMMMissing | None = None,
) -> npt.NDArray[np.float64]:
    """
    Expected Squared Distance as proposed by Eirola et al. See http://dx.doi.org/10.1016/j.neucom.2013.07.050
    Using Mixture of Gaussians for estimation.
    The GMM that minimises the corrected AIC gets selected (from all GMMs with k in [min_k, max_k]).
    Algorithm implemented as described in section 3.

    Additional parameters

    :param min_k: Minimum number of Gaussian components to try for GMM.
    :param max_k: Maximum number of Gaussian components to try for GMM.
    :param max_iter: Maximum allowed iterations for each GMM fitting.
    :param gmm: Pre-computed Gausssian Mixture Model to use instead of attempting multiple models
    """
    n_observations, n_features = data.shape

    ##
    ## Steps 1-2, 3: Fit models and get AICc and LL for each model;
    ## get model with minimal AICC
    ##

    used_model = gmm
    if used_model is None:
        used_model = get_best_gmm(min_k, max_k, max_iter, "AICc", data)

    ##
    ## Step 3: Get conditional means/covars
    ##

    s = np.full(
        (n_observations), 0.0
    )  # Holds the summed variance terms for each observation (equation (11))
    cond_mean = np.full(data.shape, 0.0)
    estimated_covars = used_model.covariances_
    estimated_means = used_model.mu_
    estimated_resp = used_model.predict_proba()  # This is t

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
            cond_mean[i] = imp_x_i

            imp_Sigma_i += t_ik * (imp_Sigma_ik + imp_x_ik @ imp_x_ik.transpose())

        imp_Sigma_i -= imp_x_i @ imp_x_i.transpose()

        s[i] = np.linalg.trace(imp_Sigma_i)

    ##
    ## Step 4: Esimate distances
    ##

    pdist = scipy.spatial.distance.pdist(cond_mean, "sqeuclidean")

    for i in range(n_observations):
        for j in range(i + 1, n_observations):
            pdist[n_observations * i + j - ((i + 2) * (i + 1)) // 2] += (
                s[i] + s[j]
            )  # Apply correction

    return pdist


def eirola_esd_mvn(
    data: npt.NDArray[np.float64], max_iter: int = 200
) -> npt.NDArray[np.float64]:
    """
    Expected Squared Distance as proposed by Eirola et al.
    Using Multivariate Normal Distributions for estimation.
    See https://doi.org/10.1016/j.ins.2013.03.043

    Additional parameters

    :param max_iter: Maximum allowed iterations for MVN fitting.
    """
    return eirola_esd_gmm(data, min_k=1, max_k=1, max_iter=max_iter)


_mapping: dict[DistFunName, DistFun] = {
    "dixon_pds_euclidean": dixon_pds_euclidean,
    "dixon_pds_sqeuclidean": dixon_pds_sqeuclidean,
}

_pdist_mapping: dict[DistFunName, PDistFun] = {
    "eirola_esd_mvn": eirola_esd_mvn,
    "eirola_esd_gmm": eirola_esd_gmm,
    "mesquita_eed": mesquita_eed,
}

def _completecase(dist: DistFun) -> DistFun:
    """
    Returns a distance function that applies the passed function only
    to the subset of both vector's features that are pairwise complete.
    Generally not recommended as the resulting metric is not adjusted for
    loss of dimensionality.
    """

    def wrapped(a: Vector, b: Vector, **kwargs: Any):
        nan_mask = np.isnan(a) | np.isnan(b)

        if len(a) == np.sum(nan_mask):
            raise DistanceError(
                "The two input vectors have no overlapping observed features, cannot estimate distance using complete case analysis."
            )
        masked_a = a[~nan_mask]
        masked_b = b[~nan_mask]
        d = dist(masked_a, masked_b, **kwargs)

        return d

    return wrapped

def get_preferred_pdist_implementation(
    distance: DistFunName | DistFun,
    distance_args: dict[str, Any] | None,
    use_completecase_analysis: bool,
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
        if use_completecase_analysis:
            distance = _completecase(distance)
        return lambda mat: scipy.spatial.distance.pdist(mat, distance)

    # Scipy supported distance function by name
    if distance in list(typing.get_args(ScipySupportedDist)):
        if use_completecase_analysis:
            distfun = getattr(scipy.spatial.distance, distance)
            distfun = _completecase(distfun)
            return lambda mat: scipy.spatial.distance.pdist(
                mat, distfun, **distance_args
            )
        return lambda mat: scipy.spatial.distance.pdist(mat, distance, **distance_args)

    # nandist supported distance function by name
    if distance.startswith("nandist_"):
        metric = distance[8:]
        return lambda mat: squareform(
            nandist.cdist(mat, mat, metric, **distance_args), checks=False
        )

    # CHM supported distance function by name
    if distance in _mapping.keys():
        fun_with_args = lambda a, b: _mapping[distance](a, b, **distance_args)
        return lambda mat: scipy.spatial.distance.pdist(mat, fun_with_args)

    # CHM supported pdist-compatible distance function by name
    if distance in _pdist_mapping.keys():
        return lambda mat: _pdist_mapping[distance](mat, **distance_args)

    raise ValueError(f"Distance function {distance} not supported")
