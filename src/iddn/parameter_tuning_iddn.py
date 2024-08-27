"""The module for tuning hyperparameters in iDDN

We use parallel computing and several tricks to make it feasible for parameter tuning in larger data

A bette approach is to manually choose a set of lambda1 and select the one that leads to a reasonable network.
By utilizing the prior knowledge, it is more likely to obtain the network that is usable.
"""

import numpy as np
from iddn import iddn
from ddn3 import tools
from joblib import Parallel, delayed

from ddn3.parameter_tuning import (
    get_lambda1_mb,
    get_lambda2_bai,
    get_lambda_one_se_1d,
    plot_error_1d,
    plot_error_2d,
)


def cv_2d(
    dat1,
    dat2,
    dep_mat,
    n_cv=5,
    ratio_val=0.2,
    lambda1_lst=np.arange(0.05, 1.05, 0.05),
    lambda2_lst=np.arange(0.025, 0.525, 0.025),
    cores=8,
    n_max=100,
    iddn_method="resi",
):
    """Cross validation by grid search lambda1 and lambda2

    To estimate the validation error, we estimate the coefficient of each node on the training set based on the
    estimated network topology. Then for each node in the validation set, we try to use its neighbors to explain the
    signal in that node. The portion of unexplained signal in all nodes is defined as the validation error.

    Although this function supports 2D grid search of hyperparameters, we can also do 1D search.
    We simply need to provide a single value for `lambda1_lst` or `lambda2_lst`.

    Let `K` be the number of CV repeats, `L1` the number of lambda1 values, `L2` the number of lambda2 values.
    `N` is the sample size, `P` is the feature number.

    Parameters
    ----------
    dat1 : (N,P) array_like
        Data for condition 1
    dat2 : (N,P) array_like
        Data for condition 1
    dep_mat : (P, P) array_like
        Constraints (dependency) matrix of iDDN.
    n_cv : int
        Number of repeats. Can be as large as you like, as we re-sample each time.
    ratio_val : float
        Ratio of iddn_data for validation. The remaining is used for training.
    lambda1_lst : array_like
        Values of lambda1 for searching
    lambda2_lst : array_like
        Values of lambda2 for searching
    cores : int
        Number of cores used in parallel computing. Should not exceed the number of cores in the computer.
    n_max : int
        The maximum number of edges allowed for each node during parameter search.
        As the regression step is time-consuming, limit the edge number will be beneficial.
        Besides, often we would prefer a sparse network, so the limit here will not influence accuracy much.
        Note that this limit only occurs for parameter tuning, and iDDN does not have this.
    iddn_method : str
        iDDN optimization method, can be `resi` or `corr`.

    Returns
    -------
    val_err : (K, L1, L2) array_like
        The validation error for each lambda1 and lambda2 combination.

    """
    val_err = np.zeros((n_cv, len(lambda1_lst), len(lambda2_lst)))
    n_node = dat1.shape[1]
    n1 = dat1.shape[0]
    n2 = dat2.shape[0]
    n1_val = int(n1 * ratio_val)
    n1_train = n1 - n1_val
    n2_val = int(n2 * ratio_val)
    n2_train = n2 - n2_val

    for n in range(n_cv):
        # Randomly divide data into training and validation set
        msk1 = np.zeros(n1)
        msk1[np.random.choice(n1, n1_train, replace=False)] = 1
        msk2 = np.zeros(n2)
        msk2[np.random.choice(n2, n2_train, replace=False)] = 1
        g1_train = tools.standardize_data(dat1[msk1 > 0])
        g1_val = tools.standardize_data(dat1[msk1 == 0])
        g2_train = tools.standardize_data(dat2[msk2 > 0])
        g2_val = tools.standardize_data(dat2[msk2 == 0])

        for i, lambda1 in enumerate(lambda1_lst):
            for j, lambda2 in enumerate(lambda2_lst):
                lambda1_mat = np.copy(dep_mat) * lambda1
                lambda2_mat = np.copy(dep_mat) * lambda2
                print(n, lambda1, lambda2, "s0")
                if cores > 1:
                    g_beta_est = iddn.iddn_parallel(
                        g1_train,
                        g2_train,
                        lambda1=lambda1_mat,
                        lambda2=lambda2_mat,
                        dep_mat=dep_mat,
                        mthd=iddn_method,
                        n_process=cores,
                    )
                else:
                    g_beta_est = iddn.iddn(
                        g1_train,
                        g2_train,
                        lambda1=lambda1_mat,
                        lambda2=lambda2_mat,
                        dep_mat=dep_mat,
                        mthd=iddn_method,
                    )
                g1_net_est = tools.get_net_topo_from_mat(g_beta_est[0])
                g2_net_est = tools.get_net_topo_from_mat(g_beta_est[1])

                # Calculate regression coefficients
                # print(n, i, j, "s1")
                g1_coef = calculate_regression(
                    g1_train, g1_net_est * g_beta_est[0], cores=cores, n_max=n_max
                )
                g1_coef[np.arange(n_node), np.arange(n_node)] = 0
                g2_coef = calculate_regression(
                    g2_train, g2_net_est * g_beta_est[1], cores=cores, n_max=n_max
                )
                g2_coef[np.arange(n_node), np.arange(n_node)] = 0

                # residual errors
                rec_ratio1 = np.linalg.norm(
                    g1_val @ g1_coef.T - g1_val
                ) / np.linalg.norm(g1_val)
                rec_ratio2 = np.linalg.norm(
                    g2_val @ g2_coef.T - g2_val
                ) / np.linalg.norm(g2_val)

                if rec_ratio1 > 2.0 or rec_ratio2 > 2.0:
                    print("High reconstruction error")
                val_err[n, i, j] = (rec_ratio1 + rec_ratio2) / 2

    return val_err


def calculate_regression(data, topo_est, cores=8, n_max=100):
    """Linear regression based on estimated network topology

    For each variable, use all its neighbors as predictors and find the regression coefficients.
    This is calculated for each condition.

    This is an example of regression operation:
    >>> x = np.array([[-1,-1,1,1.0], [1,1,-1,-1]]).T
    >>> y = np.array([1,1,-1,-1.0])
    >>> out = np.linalg.lstsq(x, y, rcond=None)
    >>> out[0]

    Let `P` be the number of features.

    Parameters
    ----------
    data : (N,P) array_like
        All iddn_data
    topo_est : (P,P) array_like
        Estimated adjacency matrix
    cores : int
        Number of cores used for parallel computing
    n_max : (P) array_like or int
        The maximum number of edges allowed for each node during parameter search.
        We can use the same value for all nodes, or assign different values for each node.
        See also``cv_2d`` function.

    Returns
    -------
    g_asso : (P,P) array_like
        Regression coefficients

    """
    n_fea = data.shape[1]
    if type(n_max) == int:
        n_max_lst = np.zeros(n_fea) + n_max
    else:
        n_max_lst = n_max

    g_asso = np.zeros((n_fea, n_fea), dtype=np.double)
    topo_est = np.abs(topo_est)

    if cores >= 1:
        out = Parallel(n_jobs=cores)(
            delayed(_regression_one_node)(
                data=data,
                topo_now=topo_est[i],  # NOTE: better to use columns
                i=i,
                n_max=int(n_max_lst[i]),
            )
            for i in range(n_fea)
        )
    else:
        # Use single core, for debug
        out = []
        for i in range(n_fea):
            if i % 10 == 0:
                print("fea", i)
            out0 = _regression_one_node(
                data=data,
                topo_now=topo_est[i],
                i=i,
                n_max=n_max_lst[i],
            )
            out.append(out0)

    for i in range(n_fea):
        if len(out[i][1]) > 0:
            g_asso[i, out[i][1]] = out[i][0]

    return g_asso


def _regression_one_node(data, topo_now, i, n_max, clip_thr=1.0):
    """Perform linear regression for one node

    The predictors are its neighbors specified by topo_now.

    Parameters
    ----------
    data : (N,P) array_like
        Input data
    topo_now : (P) array_like
        Binary mask indicating whether a node is a predictor.
    i : int
        Index of current node (response variable)
    n_max : (P) array_like or int
        The maximum number of edges allowed for each node during parameter search.
    clip_thr : float or int
        Clip too large regression coefficient to suppress overfitting a little bit

    Returns
    -------
    out : (P) array_like
        Estimated regression coefficients
    pred_idx : array_like
        Index of predictors, for debug

    """
    pred_idx = np.where(topo_now > 0)[0]

    # If there are too many predictors, choose some top ones
    if len(pred_idx) > n_max:
        pred_idx = np.argsort(-topo_now)[:n_max]
    if len(pred_idx) == 0:
        out = pred_idx
    else:
        y = data[:, i]
        x = data[:, pred_idx]
        out = np.linalg.lstsq(x, y, rcond=None)[0]

        # Reduce overfitting a bit
        out[out < -clip_thr] = clip_thr
        out[out > clip_thr] = clip_thr

        # if np.max(np.abs(out)) > 10:
        #     print("wrong")

    return out, pred_idx
