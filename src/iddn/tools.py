"""Utility functions of iDDN
"""

import numpy as np
import pandas as pd
from iddn import iddn
from ddn3 import tools, performance


def evaluate_metrics(net_est: np.ndarray, net_gt: np.ndarray):
    """Calculate the recall, precision, and F1 scores for iDDN estimates given ground truth

    Parameters
    ----------
    net_est : array_like
        Estimated network dependency matrix. The weights will be binarized.
    net_gt : array_like
        Ground truth network dependency matrix. The weights will be binarized.

    Returns
    -------
    None

    """
    res = performance.get_error_measure_two_theta(net_est, net_gt)
    recall = res[2]
    precision = res[4]
    f1 = 2 * recall * precision / (recall + precision)
    print(f"recall={recall}, precision={precision}, F1={f1}")


def get_comm_diff_network(out_iddn):
    """Find the common and differential network from iDDN estimates

    Parameters
    ----------
    out_iddn : (2,P,P) arraylike
        The raw output of iDDN. P is the number of features.

    Returns
    -------
    Common network and differential network matrices

    """
    return tools.get_common_diff_net_topo(out_iddn)


def iddn_basic_pipeline(dat1, dat2, dep_mat=None, lambda1=0.2, lambda2=0.05):
    """A convenient pipeline for iDDN

    Let `P` be the number of features (like genes or any molecules).
    `N1` is the number of samples in condition1, `N2` in conditions.

    The data will be standardized by iDDN, so users do not need to standardize it.
    The data from two conditions can have different sample size, but the feature number must be the same.

    Parameters
    ----------
    dat1 : (N1,P) array_like
        The data in condition 1.
    dat2 : (N2,P) array_like
        The data in condition 2.
    dep_mat : (P,P) array_like
        Constraints or dependency matrix
    lambda1 : float
        The penalty for overall sparsity, from 0 to 1.
    lambda2 : float
        The penalty for the discrepancies between the networks under two conditions

    Returns
    -------
    A dictionary containing results. `comm` (P by P) is the estimated common network,
    `diff` (P by P) the  differential network.
    `g1` (P by P) is the network under the first condition. `g2` (P by P) is the network under the second condition.
    `out_iddn` (2 by P by P) is the raw output of iDDN.

    """
    n_node = dat1.shape[1]
    if dep_mat is None:
        dep_mat = np.ones((n_node, n_node))
    lambda1_mat = np.copy(dep_mat) * lambda1
    lambda2_mat = np.copy(dep_mat) * lambda2
    out_iddn = iddn.iddn(
        dat1,
        dat2,
        lambda1=lambda1_mat,
        lambda2=lambda2_mat,
        dep_mat=dep_mat,
    )
    # Raw output to the network under each condition
    g1_est = tools.get_net_topo_from_mat(out_iddn[0])
    g2_est = tools.get_net_topo_from_mat(out_iddn[1])

    # Common and differential networks from raw output
    comm_est, diff_est = get_comm_diff_network(out_iddn)
    return dict(comm=comm_est, diff=diff_est, g1=g1_est, g2=g2_est, out_iddn=out_iddn)


def iddn_output_to_csv(out_iddn, node_names):
    """Convert iDDN results to Pandas data frames

    This is useful for sharing the results, as well as visualization.
    Each row of the data frame is one edge.
    There are four columns: the first node in an edge, the second node in an edge,
    the condition at which the edge exist, the weight, and the color of that edge.
    For common networks, the conditions are all set as 0.
    For differential networks, if an edge only exists in the first condition, the condition is set as 0.
    If an edge only exists in the second condition, the condition is set as 1.

    Let `P` be the number of features.

    Parameters
    ----------
    out_iddn : (2,P,P) array_like
        The raw output of iDDN.
    node_names : (P) array_like
        The list of node names to output.

    Returns
    -------
    df_edge_comm : pd.DataFrame
        A Pandas data frame for common network.
    df_edge_diff : pd.DataFrame
        A Pandas data frame for differential network.
    nodes_show_comm : array_like
        The list of node names that is present in the estimated common network.
        In other words, we only keep nodes that has at least one edge with other nodes.
    nodes_show_diff : array_like
        The list of node names that is present in the estimated common network.

    """
    g1_org = out_iddn[0]
    g2_org = out_iddn[1]
    g12_org = (g1_org + g2_org) / 2  # Weight for common edges
    g1_est = tools.get_net_topo_from_mat(g1_org)
    g2_est = tools.get_net_topo_from_mat(g2_org)

    # Find common and differential connectivity matrices
    # take only the upper part to prevent double counting
    con01 = g1_est + g2_est
    comm_net = np.triu(1 * (con01 == 2))
    diff_net = np.triu(1 * (con01 == 1))
    diff1 = 1 * ((g1_est + diff_net) == 2)
    diff2 = 1 * ((g2_est + diff_net) == 2)

    # collect edges for common network
    df_edge_comm, nodes_show_comm = collect_edges(comm_net, g12_org, node_names)

    # collect for differential network. Once for each condition, then combine them.
    edges_diff1, nodes_show_diff1 = collect_edges(
        diff1, g1_org, node_names, group=0, color_in="blue"
    )
    edges_diff2, nodes_show_diff2 = collect_edges(
        diff2, g2_org, node_names, group=1, color_in="red"
    )
    df_edge_diff = pd.concat([edges_diff1, edges_diff2], ignore_index=True)
    nodes_show_diff = list(set(nodes_show_diff1 + nodes_show_diff2))
    return df_edge_comm, df_edge_diff, nodes_show_comm, nodes_show_diff


def collect_edges(
    conn_mat,
    wt_mat,
    node_names,
    group=0,
    color_in="blue",
):
    """Convert the adjacency matrix to Pandas data frame

    For differential network, call this function twice, once for each condition, and then combine them.

    Let `P` be the number of features.

    Parameters
    ----------
    conn_mat : (P,P) array_like
        The adjacency or connectivity matrix
    wt_mat : (P,P) array_like
        Similar to conn_mat, but with weights
    node_names : (P) array_like
        The names of all nodes
    group : int
        The index of condition, can be 0 or 1
    color_in : str
        The color for this data frame.

    Returns
    -------
    df_edge : pd.DataFrame
        A Pandas data frame for the network.
    nodes_show : array_like
        The list of node names that is present in the estimated network.

    """

    # Find the edges from the connectivity matrix
    conn_mat = np.triu(conn_mat)
    ind = np.where(conn_mat > 0)
    x0 = ind[0]
    y0 = ind[1]

    # The corresponding weights and names
    weight = wt_mat[x0, y0]
    node_names_arr = np.array(node_names)
    node_x0 = node_names_arr[x0]
    node_y0 = node_names_arr[y0]
    gene1 = list(node_x0)
    gene2 = list(node_y0)
    condition = np.zeros(len(list(node_x0)), dtype=int) + group
    color = [color_in for _ in range(len(gene1))]

    df_edge = pd.DataFrame(
        dict(gene1=gene1, gene2=gene2, condition=condition, weight=weight, color=color)
    )
    nodes_show = list(set(list(node_x0) + list(node_y0)))

    return df_edge, nodes_show
