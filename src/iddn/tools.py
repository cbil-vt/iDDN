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


def generate_sample_two_conditions(g1_cov, g2_cov, n1, n2):
    """Generate multivariante normal samples for two conditions

    Let P be the number of features.

    Parameters
    ----------
    g1_cov : array_like
        Covariance matrix for condition 1
    g2_cov : array_like
        Covariance matrix for condition 2
    n1 : int
        Number of samples for condition 1
    n2 : int
        Number of samples for condition 2

    Returns
    -------
    dat1 : ndarray
        Generated samples for condition 1. Shape (n1, P)
    dat2 : ndarray
        Generated samples for condition 2. Shape (n2, P)
    """
    dat1 = generate_mvn_samples(g1_cov, n1)
    dat2 = generate_mvn_samples(g2_cov, n2)
    return dat1, dat2


def make_precision_positive_definite(mat_adj, v=0.3, u=0.1):
    n_node = mat_adj.shape[0]
    x_eigval, _ = np.linalg.eigh(mat_adj * v)
    mat_prec = mat_adj * v + (np.abs(np.min(x_eigval)) + u) * np.eye(n_node)
    mat_prec_eigval, _ = np.linalg.eigh(mat_prec)
    print(np.min(mat_prec_eigval))
    return mat_prec


def get_covariance_scaled_from_precision(prec_mat_in):
    """Create covariance from temporary precision matrix

    Each variable now have unit variance.
    We also provide the corresponding precision matrix.

    We follow [Peng 2009] and do not use the d_ij term as the JGL paper.

    Parameters
    ----------
    prec_mat_in : ndarray
        Input precision matrix

    Returns
    -------
    cov_mat : ndarray
        Modified covariance matrix
    prec_mat : ndarray
        Corresponding precision matrix
    """
    cov_mat_temp = np.linalg.inv(prec_mat_in)
    d_sqrt = np.sqrt(np.diag(1 / np.diagonal(cov_mat_temp)))
    cov_mat = d_sqrt @ cov_mat_temp @ d_sqrt
    prec_mat = np.linalg.inv(cov_mat)

    return cov_mat, prec_mat


def get_info_from_precision(omega1, omega2):
    """Get covariance matrices and adjacency matrices from two precisions matrices

    Parameters
    ----------
    omega1 : array_like
        The precision matrix for condition 1
    omega2 : array_like
        The precision matrix for condition 2

    Returns
    -------
    g1_cov : ndarray
        Covariance matrix for condition 1
    g2_cov : ndarray
        Covariance matrix for condition 2
    comm_gt : ndarray
        Adjacency matrix of common network
    diff_gt : ndarray
        Adjacency matrix of differential network
    """
    g1_cov, _ = get_covariance_scaled_from_precision(omega1)
    g2_cov, _ = get_covariance_scaled_from_precision(omega2)
    comm_gt, diff_gt = get_common_diff_adjacency([omega1, omega2])
    return g1_cov, g2_cov, comm_gt, diff_gt


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
