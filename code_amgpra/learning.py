import numpy as np
from scipy.stats import norm

def lf_eff(mu, sigma, a=0.0):
    """Implementation of the Expected Feasible Function"""
    sigma = np.maximum(sigma, 1e-12) # avoid division by zero
    eps = 2.0 * sigma
    a_plus = a + eps
    a_minus = a - eps


    z = (a - mu) / sigma
    z_plus = (a_plus - mu) / sigma
    z_minus = (a_minus - mu) / sigma

    term1 = (mu - a) * (2 * norm.cdf(z) - norm.cdf(z_minus) - norm.cdf(z_plus))
    term2 = -sigma * (2 * norm.pdf(z) - norm.pdf(z_minus) - norm.pdf(z_plus))
    term3 = 2 * sigma * (norm.cdf(z_plus) - norm.cdf(z_minus))

    return term1 + term2 + term3

def lf_u_m(mu, sigma):
    """Implementation of the Modified U learning function"""
    sigma = np.maximum(sigma, 1e-12)
    return sigma / np.exp(np.abs(mu))