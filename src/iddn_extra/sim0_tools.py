import numpy as np
import networkx as nx


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
    return x


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


def make_precision_positive_definite(mat_wt_in, u=0.1):
    n_node = mat_wt_in.shape[0]
    x_eigval, _ = np.linalg.eigh(mat_wt_in)
    mat_prec = mat_wt_in + (np.abs(np.min(x_eigval)) + u) * np.eye(n_node)
    mat_prec_eigval, _ = np.linalg.eigh(mat_prec)
    print(np.min(mat_prec_eigval))
    return mat_prec


def make_precision_positive_definite_constant(mat_adj, v=0.3, u=0.1):
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


def gen_graphs_preset(g_rand, ratio_diff=0.25, v=0.3, u=0.1):
    # g_rand = nx.erdos_renyi_graph(n=10, p=0.2)
    mat_adj = nx.adjacency_matrix(g_rand).todense()
    mat_adj = mat_adj * (1 - np.eye(len(mat_adj)))  # remove self loop
    # omega1, omega2 = make_two_omega_from_one_by_remove(mat_adj, ratio_diff=ratio_diff)
    omega1, omega2 = make_two_omega_from_one(mat_adj, ratio_diff=ratio_diff)
    omega1 = omega1 + np.eye(len(omega1))
    omega2 = omega2 + np.eye(len(omega2))
    g1_prec = make_precision_positive_definite_constant(omega1, v=v, u=u)
    g2_prec = make_precision_positive_definite_constant(omega2, v=v, u=u)
    return mat_adj, g1_prec, g2_prec


def make_two_omega_from_one(omega, ratio_diff=0.25, fill_value=1.0, thr=1e-4):
    n_node = len(omega)
    n_edge = round((np.sum(np.abs(omega) > thr) - n_node) / 2)

    # all valid candidates for new entries
    msk = np.tril(np.ones(n_node), k=-1)
    omega_lower = np.copy(omega)
    omega_lower[msk == 0] = 100.0
    idx_zero = np.where(np.abs(omega_lower) < thr)

    # generate two conditions
    n_diff = round(n_edge * ratio_diff)
    idx_chg = np.random.choice(len(idx_zero[0]), n_diff * 2, replace=False)
    idx_chg1 = idx_chg[:n_diff]
    idx_chg2 = idx_chg[n_diff:]

    diff1 = np.zeros((n_node, n_node))
    diff1[idx_zero[0][idx_chg1], idx_zero[1][idx_chg1]] = fill_value
    diff1 = diff1 + diff1.T
    omega1 = omega + diff1

    diff2 = np.zeros((n_node, n_node))
    diff2[idx_zero[0][idx_chg2], idx_zero[1][idx_chg2]] = fill_value
    diff2 = diff2 + diff2.T
    omega2 = omega + diff2

    return omega1, omega2


def make_two_omega_from_one_by_remove(omega, ratio_diff=0.25, thr=1e-4):
    n_node = len(omega)
    n_edge = round((np.sum(np.abs(omega) > thr) - n_node) / 2)

    # all valid candidates for removal
    omega_lower = np.tril(omega)
    idx_nz = np.where(np.abs(omega_lower) > thr)

    # generate two conditions
    n_diff = round(n_edge * ratio_diff)
    idx_chg = np.random.choice(len(idx_nz[0]), n_diff * 2, replace=False)
    idx_chg1 = idx_chg[:n_diff]
    idx_chg2 = idx_chg[n_diff:]

    diff1 = np.copy(omega)
    diff1[idx_nz[0][idx_chg1], idx_nz[1][idx_chg1]] = 0
    diff1[idx_nz[1][idx_chg1], idx_nz[0][idx_chg1]] = 0

    diff2 = np.copy(omega)
    diff2[idx_nz[0][idx_chg2], idx_nz[1][idx_chg2]] = 0
    diff2[idx_nz[1][idx_chg2], idx_nz[0][idx_chg2]] = 0

    return diff1, diff2
