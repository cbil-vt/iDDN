"""Block coordinate descent for iDDN

This module implements the block coordinate descent in [1].
Most methods are also Numba accelerated.

For iddn_data with lots of samples, consider using `bcd_corr`.
For iddn_data with lots of features, the `bcd_residual` is faster.

[1] Fu, Yi, et al. "DDN3. 0: Determining significant rewiring of biological
network structure with differential dependency networks." Bioinformatics (2024).

"""

import numpy as np
import numba


@numba.njit
def bcd_residual(
    beta_in,
    X1,
    X2,
    y1_resi,
    y2_resi,
    cur_node,
    dep_nodes,
    lambda1,
    lambda2,
    threshold,
    max_iter=10000,
):
    """BCD algorithm for iDDN using residual update strategy

    The algorithm allows warm start, which requires initial `beta_in`, `y1_resi`, and `y2_resi`.
    See `run_resi` on how to prepare these inputs.
    Denote `P` be the number features. N1 be the sample size for condition 1, and N2 for condition 2.

    Parameters
    ----------
    beta_in : (2P) array_like
        Initial beta. If initialization is not needed, use an array of all zeros
    X1 : (N1,P) array_like
        The iddn_data from condition 1
    X2 : (N2,P) array_like
        The iddn_data from condition 2
    y1_resi : (N1,1) array_like
        The initial residual signal for condition 1. If warm start is not used, it is column CurrIdx of X1.
    y2_resi : (N2,1) array_like
        The initial residual signal for condition 2. If warm start is not used, it is column CurrIdx of X2.
    cur_node : int
        Index of the current node that serve as the response variable.
    dep_nodes : (P) array_like
        Nodes that point to current node will be 1.
    lambda1 : array_like
        DDN parameter lambda1.
    lambda2 : array_like
        DDN parameter lambda2.
    threshold : float
        Convergence threshold.
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    beta : ndarray, shape 2P
        Estimated beta for two conditions on node CurrIdx
    r : int
        Number of iterations taken
    betaerr : float
        The error term

    """
    p = X1.shape[1]
    n1 = X1.shape[0]
    n2 = X2.shape[0]

    beta1 = np.copy(beta_in[:p])
    beta2 = np.copy(beta_in[p:])

    beta_err_array = np.zeros(2 * p)
    beta = np.zeros(2 * p)

    iter_count = 0
    k_last = cur_node
    while True:
        beta1_old = np.copy(beta1)
        beta2_old = np.copy(beta2)

        for i in range(p):
            if i == cur_node:
                continue
            if dep_nodes[i] == 0:
                continue
            lambda1_now = float(lambda1[i])
            lambda2_now = float(lambda2[i])

            iter_count = iter_count + 1
            k = i

            y1_resi = y1_resi - beta1[k_last] * X1[:, k_last] + beta1[k] * X1[:, k]
            y2_resi = y2_resi - beta2[k_last] * X2[:, k_last] + beta2[k] * X2[:, k]
            rho1 = np.sum(y1_resi * X1[:, k]) / n1
            rho2 = np.sum(y2_resi * X2[:, k]) / n2

            beta2d = solve2d(rho1, rho2, lambda1_now, lambda2_now)
            beta1[k] = beta2d[0]
            beta2[k] = beta2d[1]

            k_last = k

        beta_err_array[:p] = beta1 - beta1_old
        beta_err_array[p:] = beta2 - beta2_old
        beta_err = np.mean(np.abs(beta_err_array))

        if (beta_err < threshold) or (iter_count > max_iter):
            break

    beta[:p] = beta1
    beta[p:] = beta2
    return beta, iter_count, beta_err


@numba.njit
def bcd_corr(
    beta_in,
    cur_node,
    dep_nodes,
    lambda1,
    lambda2,
    corr_matrix_1,
    corr_matrix_2,
    threshold=1e-6,
    max_iter=100000,
):
    """BCD algorithm for iDDN using correlation matrix update strategy

    This approach is more suitable for larger sample sizes.
    The algorithm allows warm start, which requires initial `beta_in`.
    Denote P be the number features. N1 be the sample size for condition 1, and N2 for condition 2.

    Parameters
    ----------
    beta_in : array_like, length 2P
        Initial beta. If initialization is not needed, use an array of all zeros
    cur_node : int
        Index of the current node that serve as the response variable.
    dep_nodes : (P) array_like
        Nodes that point to current node will be 1.
    lambda1 : array_like
        DDN parameter lambda1.
    lambda2 : array_like
        DDN parameter lambda2.
    corr_matrix_1 : array_like, P by P
        Correlation matrix for condition 1
    corr_matrix_2 : array_like, P by P
        Correlation matrix for condition 2
    threshold : float
        Convergence threshold.
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    beta : ndarray, shape 2P
        Estimated beta for two conditions on node CurrIdx
    r : int
        Number of iterations taken
    delta_beta : float
        The error term

    """
    p = int(len(beta_in) / 2)
    beta1 = np.copy(beta_in[:p])
    beta2 = np.copy(beta_in[p:])
    beta_dif = np.zeros(2 * p)
    delta_beta = 0.0

    iter_count = 0
    for _ in range(max_iter):
        beta1_old = np.copy(beta1)
        beta2_old = np.copy(beta2)

        for k in range(p):
            if k == cur_node:
                continue
            if dep_nodes[k] == 0:
                continue
            lambda1_now = float(lambda1[k])
            lambda2_now = float(lambda2[k])
            iter_count = iter_count + 1

            betaBar1 = -beta1
            betaBar2 = -beta2
            betaBar1[k] = 0
            betaBar2[k] = 0
            betaBar1[cur_node] = 1
            betaBar2[cur_node] = 1

            rho1 = np.sum(betaBar1 * corr_matrix_1[:, k])
            rho2 = np.sum(betaBar2 * corr_matrix_2[:, k])

            beta2d = solve2d(rho1, rho2, lambda1_now, lambda2_now)
            beta1[k] = beta2d[0]
            beta2[k] = beta2d[1]

        beta_dif[:p] = beta1 - beta1_old
        beta_dif[p:] = beta2 - beta2_old
        delta_beta = np.mean(np.abs(beta_dif))

        if delta_beta < threshold:
            break

        # print(delta_beta)

    beta = np.zeros(2 * p)
    beta[:p] = beta1
    beta[p:] = beta2

    return beta, iter_count, delta_beta


@numba.njit
def solve2d(rho1, rho2, lambda1, lambda2):
    """Optimize for two variables corresponding to one node

    The details can be found in https://arxiv.org/abs/1203.3532

    Parameters
    ----------
    rho1 : float
        The rho from data1
    rho2 : float
        The rho from data2
    lambda1 : float
        DDN parameter lambda1
    lambda2 : float
        DDN parameter lambda2

    Returns
    -------
    beta1 : float
        Optimal coefficient for data1
    beta2 : float
        Optimal coefficient for data2

    """
    # initialize output
    area_index = 0
    beta1 = 0
    beta2 = 0

    if (
        rho2 <= (rho1 + 2 * lambda2)
        and rho2 >= (rho2 - 2 * lambda2)
        and rho2 >= (2 * lambda1 - rho1)
    ):
        area_index = 1
        beta1 = (rho1 + rho2) / 2 - lambda1
        beta2 = (rho1 + rho2) / 2 - lambda1
    if rho2 > (rho1 + 2 * lambda2) and rho1 >= (lambda1 - lambda2):
        area_index = 2
        beta1 = rho1 - lambda1 + lambda2
        beta2 = rho2 - lambda1 - lambda2
    if (
        rho1 < (lambda1 - lambda2)
        and rho1 >= -(lambda1 + lambda2)
        and rho2 >= (lambda1 + lambda2)
    ):
        area_index = 3
        beta1 = 0
        beta2 = rho2 - lambda1 - lambda2
    if rho1 < -(lambda1 + lambda2) and rho2 >= (lambda1 + lambda2):
        area_index = 4
        beta1 = rho1 + lambda1 + lambda2
        beta2 = rho2 - lambda1 - lambda2
    if (
        rho1 < -(lambda1 + lambda2)
        and rho2 < (lambda1 + lambda2)
        and rho2 >= -(lambda1 + lambda2)
    ):
        area_index = 5
        beta1 = rho1 + lambda1 + lambda2
        beta2 = 0
    if rho2 < -(lambda1 - lambda2) and rho2 >= (rho1 + 2 * lambda2):
        area_index = 6
        beta1 = rho1 + lambda1 + lambda2
        beta2 = rho2 + lambda1 - lambda2
    if (
        rho2 >= (rho1 - 2 * lambda2)
        and rho2 < (rho1 + 2 * lambda2)
        and rho2 <= (-2 * lambda1 - rho1)
    ):
        area_index = 7
        beta1 = (rho1 + rho2) / 2 + lambda1
        beta2 = (rho1 + rho2) / 2 + lambda1
    if rho2 < (rho1 - 2 * lambda2) and rho1 <= -(lambda1 - lambda2):
        area_index = 8
        beta1 = rho1 + lambda1 - lambda2
        beta2 = rho2 + lambda1 + lambda2
    if (
        rho1 <= (lambda1 + lambda2)
        and rho1 >= -(lambda1 - lambda2)
        and rho2 <= -(lambda1 + lambda2)
    ):
        area_index = 9
        beta1 = 0
        beta2 = rho2 + lambda1 + lambda2
    if rho1 > (lambda1 + lambda2) and rho2 <= -(lambda1 + lambda2):
        area_index = 10
        beta1 = rho1 - lambda1 - lambda2
        beta2 = rho2 + lambda1 + lambda2
    if (
        rho2 > -(lambda1 + lambda2)
        and rho2 <= (lambda1 - lambda2)
        and rho1 >= (lambda1 + lambda2)
    ):
        area_index = 11
        beta1 = rho1 - lambda1 - lambda2
        beta2 = 0
    if rho2 > (lambda1 - lambda2) and rho2 < (rho1 - 2 * lambda2):
        area_index = 12
        beta1 = rho1 - lambda1 - lambda2
        beta2 = rho2 - lambda1 + lambda2

    return [beta1, beta2]
