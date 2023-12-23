"""Utility function for simple simulation

These simulations are designed to illustrate the usage iDDN, and not for formal evaluation of performance.
"""

import numpy as np
import networkx as nx
from iddn import tools


class PairGraph:
    def __init__(
        self,
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

        g1_cov, g2_cov, comm_gt, diff_gt = tools.get_info_from_precision(
            g1_prec, g2_prec
        )

        self.g1_prec = g1_prec
        self.g2_prec = g2_prec
        self.g1_cov = g1_cov
        self.g2_cov = g2_cov
        self.comm_gt = comm_gt
        self.diff_gt = diff_gt
    
    def draw_sample(self, n1=200, n2=200):
        dat1 = tools.generate_mvn_samples(self.g1_cov, n1)
        dat2 = tools.generate_mvn_samples(self.g2_cov, n2)
        return dat1, dat2    


class ThreeLayerGraph:
    def __init__(
        self,
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
        g1_prec = tools.make_precision_positive_definite(mat1_adj, v=v, u=u)
        g2_prec = tools.make_precision_positive_definite(mat2_adj, v=v, u=u)

        g1_cov, g2_cov, comm_gt, diff_gt = tools.get_info_from_precision(
            g1_prec, g2_prec
        )

        self.n_node = n_node
        self.g = g_rand
        self.g1 = g1_rand
        self.g2 = g2_rand
        self.adj_mat1 = mat1_adj
        self.adj_mat2 = mat2_adj
        self.prec1 = g1_prec
        self.prec2 = g2_prec
        self.g1_cov = g1_cov
        self.g2_cov = g2_cov
        self.comm_gt = comm_gt
        self.diff_gt = diff_gt

    def get_dependency_matrix(self):
        # dependency matrix
        # each column is the feature, the nodes that go to that node is 1
        n_node = self.n_node
        dep_mat = np.zeros((n_node * 3, n_node * 3))
        dep_mat[:n_node, :n_node] = 1
        dep_mat[n_node : 2 * n_node, :n_node] = np.eye(n_node)
        dep_mat[2 * n_node :, :n_node] = np.eye(n_node)
        return dep_mat

    def get_rho_matrix(self, rho_a=0.2, rho_b=0.2):
        # hyper-parameter matrix
        # no penalty for DNA to expression edge
        n_node = self.n_node
        rho_mat = np.zeros((n_node * 3, n_node * 3))
        rho_mat[:n_node, :n_node] = rho_a
        rho_mat[n_node : 2 * n_node, :n_node] = 0
        rho_mat[2 * n_node :, :n_node] = np.eye(n_node) * rho_b
        return rho_mat

    def draw_sample(self, n1, n2):
        dat1 = tools.generate_mvn_samples(self.g1_cov, n1)
        dat2 = tools.generate_mvn_samples(self.g2_cov, n2)
        return dat1, dat2
