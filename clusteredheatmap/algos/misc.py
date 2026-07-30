import numpy as np
import numpy.typing as npt

from typing import Literal

from clusteredheatmap.types import Vector

def ecmgmm(
    data: npt.NDArray[np.float64],
    k: int,
    *,
    max_iterations: int = 400,
    tolerance: float = 1e-8,
):

    """
    Estimates weight, mean and covariance of k gaussian distributions fitted to the data
    using the Expectation Conditional Maximizaton algorithm.

    The algorithm is described in 10.1016/j.neucom.2013.07.050 (ref. [1]).
    The general structure of this implementation was influenced by the ecmnmle algorithm below,
    which is based on different sources referenced there.

    :param data: the data as a numpy matrix of shape (n_observations, n_features)
    :param k: How many Gaussians to use for the GMM
    :param max_iterations: No of iterations after which the algorithm forcely terminates (w/out convergence)
    :param tolerance: Max delta after which an increase in loglikelihood is considered a convergence

    :return: List l of length k of Tuples (pi, mu, Sigma), where for 0 <= i < k:
        l[i].pi is the mixing coefficient for the i-th Gaussian 
        l[i].mu are the means of the i-th Gaussian
        l[i].Sigma is the covariance matrix of the i-th Gaussian
    """

    invalid_observations = np.all(np.isnan(data), axis=1)
    if np.any(invalid_observations):
        raise ValueError("Illegal input: data contains at least one observation with only NaNs.")

    n_samples, n_features = data.shape

    # Masks for entire dataset
    missing = np.isnan(data)
    observed = ~missing

    def loglikelihood():
        return -np.inf

    ##
    ## Initialization
    ##

    # For each feature, we now have k Gaussian distributions, each with their own mean and covar.
    # Means are initialized by selecting random values from the observed data (as in section 2.4 in [1]).
    # As in [1], observed values are preferred over imputed values, but imputed values are used
    # if not enough values are present.
    rng = np.random.default_rng()
    estimated_feature_means = np.nanmean(data, axis=0) # only single value for each feature
    imputed_features = np.where(missing, estimated_feature_means, data).transpose()

    estimated_mean = np.full((n_features, k), 0.0)
    obs_t = observed.transpose()

    for i in range(n_features):
        weight = obs_t[i] / np.sum(obs_t[i])
        estimated_mean[i] = rng.choice(imputed_features[i], size=2, axis=0, p=weight)

    estimated_mean = estimated_mean.transpose() # Shape (k, n_features)

    # For simplicity, we're using the imputed data for an initial covariance estimation.
    estimated_covar = np.full((k, n_features, n_features), 0.0)

    for i in range(k):
        estimated_covar[i] = np.atleast_2d(np.cov(imputed_features, rowvar=True, bias=True))

    estimated_mix_coef = np.full((k), 1.0/k)

    print(estimated_mean)
    print(estimated_covar)
    print(estimated_mix_coef)

    has_converged: bool = False
    prev_ll = -np.inf
    # for _iter in range(max_iterations):

def ecmnmle(
    data: npt.NDArray[np.float64], 
    *,
    max_iterations: int = 400,
    tolerance: float = 1e-8,
):
    """
    Estimates the mean and covariance of incomplete multivariate normal data
    using the Expectation Conditional Maximizaton algorithm.

    Similar functionality as MATLAB ecmnmle https://de.mathworks.com/help/finance/ecmnmle.html

    The main reference paper was written by Sexton and Swensen https://www.econstor.eu/bitstream/10419/192227/1/dp244.pdf (ref [1])
    Additionally, I used "Statistical Analysis with Missing Data" by Little and Rubin (ref [2]).
    For simplicity, observations containing only missing values are not allowed.

    :param data: the data as a numpy matrix of shape (n_observations, n_features)
    :param max_iterations: No of iterations after which the algorithm forcely terminates (w/out convergence)
    :param tolerance: Max delta after which an increase in loglikelihood is considered a convergence
    """
    invalid_observations = np.all(np.isnan(data), axis=1)
    if np.any(invalid_observations):
        raise ValueError("Illegal input: data contains at least one observation with only NaNs.")

    n_observations, n_features = data.shape

    def loglikelihood(
        data: npt.NDArray[np.float64], 
        estimated_mean: npt.NDArray[np.float64],
        estimated_covar: npt.NDArray[np.float64],
    ):
        """
        Helper that computes the loglikelihood of our parameters on the data
        """

        ll = 0.0
        # See [2] equation 11.1.
        # Same symbol mapping as with the conditional distribution below.
        # We ignore the const term here as it doesn't contribute
        # to convergence.
        # We also move the constant 0.5 terms into the sums for cleaner code.

        for observation in data:
            observed = ~np.isnan(observation)

            y_o = observation[observed]
            mu_o = estimated_mean[observed]
            cov_oo = estimated_covar[np.ix_(observed, observed)]
            
            sign, logdet = np.linalg.slogdet(cov_oo)
            if sign < 0:
                raise Exception("Matrix is non-positive definite, shouldn't happen")

            else:
                ll += -0.5 * logdet

            try:
                inv_cov_oo = np.linalg.pinv(cov_oo, hermitian=True)
            except np.linalg.LinAlgError:
                inv_cov_oo = np.eye(np.sum(observed))
                
            diff = y_o - mu_o

            ll += -0.5 * diff.T @ inv_cov_oo @ diff

        return ll


    # Initialization like the `twostage` method MATLAB provides.
    # Estimate a mean vector by ignoring the NaNs and "impute" the data by filling with the mean.
    # Resulting mean and covar matrix are the starting point.
    estimated_mean: Vector = np.nanmean(data, axis=0)
    imputed_data = np.where(np.isnan(data), estimated_mean, data)
    estimated_covar = np.atleast_2d(np.cov(imputed_data, rowvar=False, bias=True)) # bias as per MATLAB docs

    has_converged: bool = False
    prev_ll = -np.inf
    for _iter in range(max_iterations):
        
        ##
        ## Expectation
        ## 
        ## See section 11.2.1 in [2]

        # See [2] equations 11.2 and 11.3
        sum_y = np.zeros(n_features)
        sum_yy = np.zeros((n_features, n_features))

        for observation in data:
            missing = np.isnan(observation)
            observed = ~missing

            # See [2] equations 11.4 and 11.5, we fill the nans after this
            y = observation.copy()
            cov_i = np.zeros((n_features, n_features))

            # Using the conditional distribution now
            # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
            # to get the expected value under the condition of our
            # current parameters.
            # Index 1 is equivalent to the observed values (o)
            # Index 2 is equivalent to the missing values (m)
            if np.any(missing):
                y_o = y[observed]
                mu_o = estimated_mean[observed]
                mu_m = estimated_mean[missing]

                cov_oo = estimated_covar[np.ix_(observed, observed)]
                cov_mo = estimated_covar[np.ix_(missing, observed)]
                cov_om = estimated_covar[np.ix_(observed, missing)]
                cov_mm = estimated_covar[np.ix_(missing, missing)]

                try:
                    inv_cov_oo = np.linalg.pinv(cov_oo, hermitian=True)
                    beta = cov_mo @ inv_cov_oo
                
                # Happens sometimes
                except np.linalg.LinAlgError:
                    # Same shape as regular beta but just zero
                    # So we don't consider any covar in the calculations for 
                    # this iter.
                    beta = np.zeros((np.sum(missing), np.sum(observed)))

                y_m = mu_m + beta @ (y_o - mu_o)
                c_m = cov_mm - beta @ cov_om

                y[missing] = y_m
                cov_i[np.ix_(missing, missing)] = c_m

            sum_y += y
            sum_yy += np.outer(y, y) + cov_i


        # NOTE: Both [1] and [2] describe the use of a design matrix for
        # the maximization steps. As there is no "known design matrix"
        # that we can pass to matlab (and the ESD algorithm that we do this for
        # does not mention one), we set X equal to the identity matrix.
        # This simplifies a lot of calculations down the line,
        # hence some formulas look different here than they do in the references.
        # TODO: Write down proofs for this or nah?
        new_estimated_mean = sum_y / n_observations
        new_estimated_covar = sum_yy / n_observations - np.outer(new_estimated_mean, new_estimated_mean)

        # Re-force symmetry and add jitter, else we don't converge...
        new_estimated_covar = (new_estimated_covar + new_estimated_covar.T) / 2.0
        new_estimated_covar += 1e-6 * np.eye(n_features)

        current_ll = loglikelihood(data, new_estimated_mean, new_estimated_covar)

        if abs(current_ll - prev_ll) <= tolerance:
            has_converged = True
        
        estimated_mean = new_estimated_mean
        estimated_covar = new_estimated_covar
        prev_ll = current_ll

        # print(estimated_mean)
        # print(_iter, current_ll, estimated_mean)
        # print(_iter, current_ll)

        if has_converged:
            break

    return estimated_mean, estimated_covar
