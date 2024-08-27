"""Wrapper functions for calling the BCD algorithms used in iDDN
"""

import numpy as np
from iddn import bcd


def run_resi(
    g1_data,
    g2_data,
    node,
    dep_cur,
    lambda1,
    lambda2,
    beta1_in,
    beta2_in,
    threshold,
):
    """The wrapper that calls the iDDN residual update algorithm

    Denote P be the number features. N1 be the sample size for condition 1, and N2 for condition 2.

    Parameters
    ----------
    g1_data : array_like, shape N1 by P
        The iddn_data from condition 1
    g2_data : array_like, shape N2 by P
        The iddn_data from condition 2
    node : int
        Index of the current node that serve as the response variable.
    dep_cur : (P) array_like
        The neighbors that points to current node
    lambda1 : array_like
        DDN parameter lambda1.
    lambda2 : array_like
        DDN parameter lambda2.
    beta1_in : array_like, length P
        Initial beta for condition 1. If initialization is not needed, use an array of all zeros.
    beta2_in : array_like, length P
        Initial beta for condition 2. If initialization is not needed, use an array of all zeros.
    threshold : float
        Convergence threshold.

    Returns
    -------
    beta1 : ndarray, length P
        Estimated beta for `node` in condition 1.
    beta2 : ndarray, length P
        Estimated beta for `node` in condition 2.

    """
    if np.sum(dep_cur) == 0:
        return beta1_in, beta2_in

    beta_in = np.concatenate((beta1_in, beta2_in))

    y1_resi = np.copy(g1_data[:, node])
    y2_resi = np.copy(g2_data[:, node])

    N_NODE = g1_data.shape[1]

    beta, _, _ = bcd.bcd_residual(
        beta_in,
        g1_data,
        g2_data,
        y1_resi,
        y2_resi,
        node,
        dep_cur,
        lambda1,
        lambda2,
        threshold,
    )
    beta1 = np.array(beta[:N_NODE])
    beta2 = np.array(beta[N_NODE:])

    return beta1, beta2


def run_corr(
    corr_matrix_1,
    corr_matrix_2,
    node,
    dep_cur,
    lambda1,
    lambda2,
    beta1_in,
    beta2_in,
    threshold,
):
    """The wrapper that calls the iDDN correlation matrix update algorithm

    Parameters
    ----------
    corr_matrix_1 : ndarray
        Input correlation matrix for condition 1.
    corr_matrix_2 : ndarray
        Input correlation matrix for condition 2.
    node : int
        Index of the current node that serve as the response variable.
    dep_cur : array_like
        The neighbors that points to current node
    lambda1 : float
        DDN parameter lambda1.
    lambda2 : float
        DDN parameter lambda2.
    beta1_in : array_like, length P
        Initial beta for condition 1. If initialization is not needed, use an array of all zeros.
    beta2_in : array_like, length P
        Initial beta for condition 2. If initialization is not needed, use an array of all zeros.
    threshold : float
        Convergence threshold.

    Returns
    -------
    beta1 : ndarray, length P
        Estimated beta for `node` in condition 1.
    beta2 : ndarray, length P
        Estimated beta for `node` in condition 2.
    """
    beta_in = np.concatenate((beta1_in, beta2_in))

    beta, _, _ = bcd.bcd_corr(
        beta_in,
        node,
        dep_cur,
        lambda1,
        lambda2,
        corr_matrix_1,
        corr_matrix_2,
        threshold=threshold,
        max_iter=100000,
    )

    n_node = len(beta1_in)
    beta1 = np.array(beta[:n_node])
    beta2 = np.array(beta[n_node:])

    return beta1, beta2
