"""Utility functions of DDN
"""

import numpy as np


def standardize_data(data):
    """Standadize each column of the input data"""

    dat_mean = np.mean(data, axis=0)
    dat_std = np.std(data, axis=0)
    return (data - dat_mean) / dat_std


def generate_mvn_samples(cov_mat, n_sample):
    """Generate zero mean multivariate data and standardize it.

    Parameters
    ----------
    cov_mat : ndarray
        Covariance matrix
    n_sample : int
        Sample size

    Returns
    -------
    ndarray
        Generated samples
    """
    x = np.random.multivariate_normal(np.zeros(len(cov_mat)), cov_mat, n_sample)
    return standardize_data(x)


def clean_adjacency(mat_prec, thr=1e-4):
    """Threshold the input matrxi to get adjacency matrix

    The input is also made symmetry.
    As we do not use self loop, the diagonal elements are always 0.

    Parameters
    ----------
    mat_prec : ndarray
        Input matrix. It could be a precision matrix, or a coefficient matrix.
    thr : float, optional
        Threshold to remove too small items in `mat_prec`, by default 1e-4

    Returns
    -------
    ndarray
        Adjacency matrix
    """
    N = len(mat_prec)
    x = np.copy(mat_prec)
    x[np.arange(N), np.arange(N)] = 0.0
    x = 1.0 * (np.abs(x) > thr)
    x = 1.0 * ((x + x.T) > 0)
    return x


def get_common_diff_adjacency(g_beta, thr=1e-4):
    """Calcualte common and differential network from the output of DDN

    g_beta[0] is the coefficient matrxi for condition 1
    g_beta[0] is the coefficient matrxi for condition 2
    """
    g1 = clean_adjacency(g_beta[0], thr=thr)
    g2 = clean_adjacency(g_beta[1], thr=thr)
    g_net_comm = 1.0 * ((g1 + g2) == 2)
    g_net_dif = 1.0 * (g1 != g2)
    return g_net_comm, g_net_dif
