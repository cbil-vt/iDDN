"""Utility function for simple simulation

These simulations are designed to illustrate the usage DDN, and not for formal evaluation of performance.
Only the pair graph is supported.

It also contains some helper functions for other simulation functions.

"""

import numpy as np
import networkx as nx
from iddn import tools


def create_pair_graph(
    n_node=40,
    corr=0.75,
    n_shuf=3,
):
    """Generate precision matrix for pair graph.

    For graph with N nodes, there are N/2 edges, and each node has degree one.

    Parameters
    ----------
    n_node : int, optional
        The number of nodes, by default 40
    corr : float, optional
        The value of precision matrix between two neighboring nodes, by default 0.75
    n_shuf : int, optional
        The number of edges to shuffle, by default 3

    Returns
    -------
    g1_prec : ndarray
        Precision matrix for condition 1, shape (N, N)
    g2_prec : ndarray
        Precision matrix for condition 2, shape (N, N)
    """
    n_blk = int(n_node / 2)
    edge1 = np.zeros((n_blk, 2))
    edge1[:, 0] = np.arange(n_blk)
    edge1[:, 1] = np.arange(n_blk) + n_blk

    edge2 = np.copy(edge1)
    idx_shuf = np.arange(n_blk, n_blk + n_shuf)
    edge2[: len(idx_shuf), 1] = np.concatenate([idx_shuf[[-1]], idx_shuf[:-1]])

    xx = np.array([[1, -corr], [-corr, 1]])
    # print(np.linalg.eig(xx))

    g1_prec = np.zeros((n_node, n_node))
    for e in edge1:
        e = e.astype(int)
        g1_prec[np.ix_([e[0], e[1]], [e[0], e[1]])] = xx

    g2_prec = np.zeros((n_node, n_node))
    for e in edge2:
        e = e.astype(int)
        g2_prec[np.ix_([e[0], e[1]], [e[0], e[1]])] = xx

    return g1_prec, g2_prec


def create_three_layers_graph(
    n_node=10,
    p=0.2,
    v=0.3,
    u=0.1,
    n_add_each=2,
):
    # expression edges with random graph.
    g_rand = nx.erdos_renyi_graph(n_node, p)
    d = dict()
    for i in range(n_node):
        d[i] = f"e_{i}"
    g_rand: nx.Graph = nx.relabel_nodes(g_rand, d)

    # edge with DNA copy number.
    for i in range(n_node):
        g_rand.add_edge(f"e_{i}", f"g_{i}")

    # edges with TF protein. One TF for each expression node
    for i in range(n_node):
        g_rand.add_edge(f"e_{i}", f"p_{i}")

    # create two conditions from the common `g_rand` graph
    # add two extra edges in each condition
    mat_adj = nx.adjacency_matrix(g_rand).todense()
    mat0 = np.copy(mat_adj[:n_node, :n_node])
    mat0[np.triu_indices_from(mat0)] = 1
    x, y = np.where(mat0 == 0)
    idx_sel = np.random.choice(len(x), n_add_each * 2)
    idx_sel1 = idx_sel[:n_add_each]
    idx_sel2 = idx_sel[n_add_each:]

    g1_rand = g_rand.copy()
    for i in idx_sel1:
        g1_rand.add_edge(f"e_{x[i]}", f"e_{y[i]}")
        # print(i, x[i], y[i])
    g2_rand = g_rand.copy()
    for i in idx_sel2:
        g2_rand.add_edge(f"e_{x[i]}", f"e_{y[i]}")
        # print(i, x[i], y[i])

    # make positive definite precision matrix 
    mat1_adj = nx.adjacency_matrix(g1_rand).todense()
    mat2_adj = nx.adjacency_matrix(g2_rand).todense()
    g1_prec = make_precision_positive_definite(mat1_adj, v=v, u=u)
    g2_prec = make_precision_positive_definite(mat2_adj, v=v, u=u)

    # dependency matrix
    # each column is the feature, the nodes that go to that node is 1
    dep_mat = np.zeros((n_node*3, n_node*3))
    dep_mat[:n_node,:n_node] = 1
    dep_mat[n_node:2*n_node,:n_node] = np.eye(n_node)
    dep_mat[2*n_node:,:n_node] = np.eye(n_node)

    return g1_prec, g2_prec, dep_mat, mat1_adj, mat2_adj


def make_precision_positive_definite(mat_adj, v=0.3, u=0.1):
    n_node = mat_adj.shape[0]
    x_eigval, _ = np.linalg.eigh(mat_adj * v)
    mat_prec = mat_adj * v + (np.abs(np.min(x_eigval)) + u) * np.eye(n_node)
    mat_prec_eigval, _ = np.linalg.eigh(mat_prec)
    print(np.min(mat_prec_eigval))
    return mat_prec


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
    comm_gt, diff_gt = tools.get_common_diff_adjacency([omega1, omega2])
    return g1_cov, g2_cov, comm_gt, diff_gt


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
    dat1 = tools.generate_mvn_samples(g1_cov, n1)
    dat2 = tools.generate_mvn_samples(g2_cov, n2)
    return dat1, dat2


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


def get_data_demo():
    """Generate example data for the tutorial"""
    n_node = 40
    n_sample1 = 100
    n_sample2 = 100
    n_shuf = 5
    g1_prec, g2_prec = create_pair_graph(n_node=n_node, corr=0.75, n_shuf=n_shuf)
    gene_names = [f"Gene{i}" for i in range(n_node)]

    g1_cov, _ = get_covariance_scaled_from_precision(g1_prec)
    g2_cov, _ = get_covariance_scaled_from_precision(g2_prec)
    dat1 = tools.generate_mvn_samples(g1_cov, n_sample1)
    dat2 = tools.generate_mvn_samples(g2_cov, n_sample2)

    return dat1, dat2, gene_names
