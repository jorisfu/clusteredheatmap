from typing import Literal
import numpy as np
import numpy.typing as npt

from clusteredheatmap.algos.gmm_missing.gmm import GMMMissing

InfoCriterion = Literal["AICc", "BIC"]
import logging

logger = logging.getLogger(__name__)

def bic(log_likelihood: float, n_observations: int, n_params: int) -> float:
    """
    Computes the Bayesian information criterion

    :param log_likelihood: Loglikelihood of the model on the data
    :param n_observations: Number of observations in the dataset
    :param n_params: Number of parameters estimated by the model
    """
    return np.log(n_observations) * n_params - 2.0 * log_likelihood

def corrected_aic(log_likelihood: float, n_observations: int, n_params: int) -> float:
    """
    Computes the corrected Akaike information criterion.
    See Section 2.6 in http://dx.doi.org/10.1016/j.neucom.2013.07.050 (Equation (9))

    :param log_likelihood: Loglikelihood of the model on the data
    :param k: number of Gaussians used in the model
    """
    P = n_params
    return -2 * log_likelihood + 2 * P + (2 * P * (P + 1)) / (n_observations - P - 1)

def no_of_gmm_params(k: int, n_features: int) -> int:
    """
    Number of free parameters in a GMM in the case of full, separate, covariance matrices.

    See Section 2.6 in http://dx.doi.org/10.1016/j.neucom.2013.07.050

    :param k: Number of Gaussian components used in the model
    :param n_features: Number of features for each observation
    """
    return int(k * n_features + (k - 1) + (0.5 * k * n_features * (n_features + 1)))

def get_best_gmm(min_k: int, max_k: int, max_iter: int, criterion: InfoCriterion, data: npt.NDArray[np.floating], random_state: int = 123) -> GMMMissing:
    """
    Fits GMMs with min_k <= k <= max_k components to the data and returns the best GMM
    according to the selected information criterion.

    :param min_k: Minimum number of Gaussian components to try for GMM.
    :param max_k: Maximum number of Gaussian components to try for GMM.
    :param max_iter: Maximum allowed iterations for each GMM fitting.
    :param criterion: Which information criterion to use
    :param random_state: Seed for RNG (required for GMM initialization)

    :return: GMMMissing model
    """
    n_observations, n_features = data.shape

    criterion_fun = None
    match criterion:
        case "AICc":
            criterion_fun = corrected_aic
        case "BIC":
            criterion_fun = bic

    models: list[GMMMissing] = []
    n_models = max_k - min_k + 1
    log_likelihood = np.full((n_models), 0.0)
    criterion_score = np.full((n_models), 0.0)

    for k in range(min_k, max_k + 1):
        model = GMMMissing(k, max_iter=max_iter, random_state=random_state).fit(data)
        models.append(model)

        ll = model.lower_bound_
        log_likelihood[k - min_k] = ll
        n_params = no_of_gmm_params(k, n_features)
        criterion_score[k - min_k] = criterion_fun(ll, n_observations, n_params) # Section 2.6

    used_model = models[np.argmin(criterion_score)]
    logging.info("Using GMM with " + str(used_model.n_components) + " components")
    print("Using GMM with " + str(used_model.n_components) + " components")
    print(criterion_score)
    return used_model
