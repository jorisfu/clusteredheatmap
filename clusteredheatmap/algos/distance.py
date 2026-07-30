from math import log
import numpy as np
import numpy.typing as npt
import typing
from typing import Literal

import scipy

from clusteredheatmap.algos.gmm_missing.gmm import GMMMissing
from clusteredheatmap.algos.misc import ecmnmle
from clusteredheatmap.types import DistFun, Vector, PDistFun

ScipySupportedDist = Literal['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalsneath', 'sqeuclidean', 'yule']

ChmSupportedDist = Literal["dixon_pds_euclidean", "dixon_pds_sqeuclidean", "mesquita_eed", "eirola_esd_mvn", "eirola_esd_gmm"]

DistFunName = ScipySupportedDist | ChmSupportedDist

def _dixon_pds_euclidean(a: Vector, b: Vector, sqrt: bool = True) -> np.float64:
    """
    Partial Distance Strategy as proposed by Dixon. See "Pattern Recognition with Partly Missing Data" by John K. Dixon.
    """
    nan_mask = np.isnan(a) | np.isnan(b)
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

def mesquita_eed(a: Vector, b: Vector) -> np.float64:
    """
    Expected Euclidean Distance as proposed by Mesquita et al. See http://dx.doi.org/10.1016/j.neucom.2016.12.081

    Assumes distances are Nakagami-distributed. Data distribution modeled via a Gaussian mixture distribution.
    """

    # TODO
    return np.float64(0.0)

def eirola_esd_gmm(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Expected Squared Distance as proposed by Eirola et al. See [TODO].
    Using Mixture of Gaussians for esimtation.
    Algorithm implemented as described in section 3.
    """
    n_observations, n_features = data.shape

    def corrected_aic(log_likelihood: np.floating, k: int) -> np.floating:
        """
        See section 2.6. Computes the corrected Akaike information
        criterion.

        :param log_likelihood: log_likelihood of the model on the data
        :param k: number of Gaussians used in the model
        """
        N, d = data.shape
        P = k * d + (k - 1) + (0.5 * k * d * (d+1))
        try:
            return -2 * log_likelihood + 2 * P + (2*P*(P+1)) / (N - P - 1)
        # TODO: Investigate this fr
        except ZeroDivisionError:
            return 0.0

    # TODO: Maybe make stuff like this actually parametrizable
    max_k = 3

    # models: list[GMMMissing] = []
    # log_likelihood = np.full((max_k), 0.0)
    # aic = np.full((max_k), 0.0)
    #
    # for k in range(1, max_k+1):
    #     model = GMMMissing(k, max_iter=40)
    #     model.fit(data)
    #     models.append(model)
    #     ll = model.lower_bound_
    #     log_likelihood[k-1] = ll
    #     aic[k-1] = corrected_aic(ll, k)

    # print(log_likelihood)
    # print(aic)

    # used_model = models[1] # k = 3 TODO dynamic
    used_model = GMMMissing(2, max_iter=100).fit(data)

    s = np.full((n_observations), 0.0)
    imputed_data = used_model.X_final_
    estimated_covars = used_model.covariances_
    estimated_means = used_model.mu_
    estimated_mixcoefs = used_model.alpha_
    t = used_model.predict_proba()
    missing = np.isnan(data)
    observed = ~missing

    for i in range(n_observations):
        obs_i = observed[i]
        mis_i = missing[i]
        
        imp_x_i = np.full((n_features), 0.0)
        imp_Sigma_i = np.full((n_features, n_features), 0.0)

        for k in range(used_model.n_components):
            mu_k = estimated_means[k]
            t_ik = t[i][k]
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

            imp_Sigma_i += t_ik * (imp_Sigma_ik + imp_x_ik @ imp_x_ik.transpose())

        # imp_Sigma_i -= used_model.n_components * (imp_x_i @ imp_x_i.transpose())
        imp_Sigma_i -= (imp_x_i @ imp_x_i.transpose())

        s[i] = np.linalg.trace(imp_Sigma_i)

    pdist = scipy.spatial.distance.pdist(imputed_data, "sqeuclidean")

    print(pdist)

    for i in range(n_observations):
        for j in range(i+1, n_observations):
            pdist[n_observations * i + j - ((i + 2) * (i + 1)) // 2] += (s[i] + s[j])
    
    print(pdist)

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
        for j in range(i+1, n_observations):
            pdist[n_observations * i + j - ((i + 2) * (i + 1)) // 2] += (s_squared[i] + s_squared[j])

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
) -> PDistFun:
    """
    From a given name of a distance function to use or a distance function, returns
    a callable function compatible with pdist to create 
    a condensed distance matrix from an array of observation
    vectors.
    """

    if callable(distance):
        return lambda mat: scipy.spatial.distance.pdist(mat, distance)

    if distance in list(typing.get_args(ScipySupportedDist)):
        return lambda mat: scipy.spatial.distance.pdist(mat, distance)
    elif distance in _mapping.keys():
        return lambda mat: scipy.spatial.distance.pdist(mat, _mapping[distance])
    elif distance in _pdist_mapping.keys():
        return _pdist_mapping[distance]

    raise ValueError(f"Distance function {distance} not supported")

