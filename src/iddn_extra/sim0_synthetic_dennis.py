import numpy as np
import networkx as nx


def three_layer_topo_dennis(
    n_gene=15,
    n_tf=5,
    n_cis=1,
    use_fixed_wt=True,
    n_gg_pair_add=5,
    n_gt_pair_add=4,
):
    # common graph
    G = nx.Graph()
    a = [f"G{i}" for i in range(1, n_gene + 1)]
    b = [f"E{i}" for i in range(1, n_gene + 1)]
    c = [f"P{i}" for i in range(1, n_tf + 1)]
    G.add_nodes_from(a + b + c)

    # Gene dosage
    for i in range(1, n_gene + 1):
        G.add_edge(f"G{i}", f"E{i}", s=1)

    # cis correlation
    msk = np.arange(1, n_gene + 1)
    # n_cis = 2
    for i in range(1, n_gene + 1):
        msk1 = msk[msk != i]
        idx = msk1[np.random.choice(len(msk1), n_cis, replace=False)]
        for idx0 in idx:
            G.add_edge(f"G{i}", f"E{idx0}", s=2)
            # print(f"G{i}", f"E{idx0}")

    # cnv_cis_tgt = [4, 6, 2, 8, 1, 2, 14, 4, 15, 15, 10, 9, 7, 9, 10]
    # for i in range(1, n_gene + 1):
    #     G.add_edge(f"G{i}", f"E{cnv_cis_tgt[i-1]}", s=2)

    # gene gene
    gg_pairs = [
        [1, 4],
        [2, 4],
        [3, 4],
        [4, 8],
        [5, 8],
        [6, 8],
        [8, 14],
        [9, 15],
        [10, 14],
        [11, 14],
        [13, 14],
    ]
    for p in gg_pairs:
        G.add_edge(f"E{p[0]}", f"E{p[1]}", s=3)

    # valid gene-gene pairs to add
    gene_gene_pairs_more = []
    for i in range(1, n_gene + 1):
        for j in range(i + 1, n_gene + 1):
            if [i, j] in gg_pairs or [j, i] in gg_pairs:
                continue
            gene_gene_pairs_more.append([i, j])

    # TF gene
    tf_tgt = [
        [1, 2, 14],
        [6, 9, 13],
        [2, 15],
        [7, 8, 13],
        [3, 4, 6, 7, 11],
    ]
    for i in range(n_tf):
        p = tf_tgt[i]
        for p0 in p:
            G.add_edge(f"P{i+1}", f"E{p0}", s=4)

    # valid gene-tf pairs to add
    gene_tf_pairs_more = []
    for i in range(n_tf):
        for j in range(1, n_gene + 1):
            if j in tf_tgt[i]:
                continue
            gene_tf_pairs_more.append([i + 1, j])

    if use_fixed_wt:
        mat_weight = get_weight_matrix_fixed(G)
    else:
        mat_weight = get_weight_matrix(G)

    idx = np.random.choice(len(gene_gene_pairs_more), n_gg_pair_add*2, replace=False)
    gg_idx1 = idx[:n_gg_pair_add]
    gg_idx2 = idx[n_gg_pair_add:]

    idx = np.random.choice(len(gene_tf_pairs_more), n_gt_pair_add*2, replace=False)
    gt_idx1 = idx[:n_gt_pair_add]
    gt_idx2 = idx[n_gt_pair_add:]

    # condition 1
    G1 = G.copy()
    for i in gg_idx1:
        a = f"E{gene_gene_pairs_more[i][0]}"
        b = f"E{gene_gene_pairs_more[i][1]}"
        G1.add_edge(a, b, s=3)
        # print(a, b)
    for i in gt_idx1:
        a = f"P{gene_tf_pairs_more[i][0]}"
        b = f"E{gene_tf_pairs_more[i][1]}"
        G1.add_edge(a, b, s=4)
        # print(a, b)

    # G1.add_edge("E1", "E5", s=3)
    # G1.add_edge("E5", "E13", s=3)
    # G1.add_edge("E7", "E8", s=3)
    # G1.add_edge("E12", "E14", s=3)
    # G1.add_edge("E9", "E14", s=3)

    # G1.add_edge("P1", "E3", s=4)
    # G1.add_edge("P1", "E15", s=4)
    # G1.add_edge("P2", "E14", s=4)
    # G1.add_edge("P3", "E3", s=4)

    if use_fixed_wt:
        mat_weight1 = get_weight_matrix_fixed(G1)
    else:
        mat_weight1 = get_weight_matrix(G1)

    # condition 2
    G2 = G.copy()
    for i in gg_idx2:
        a = f"E{gene_gene_pairs_more[i][0]}"
        b = f"E{gene_gene_pairs_more[i][1]}"
        G2.add_edge(a, b, s=3)
        # print(a, b)
    for i in gt_idx2:
        a = f"P{gene_tf_pairs_more[i][0]}"
        b = f"E{gene_tf_pairs_more[i][1]}"
        G2.add_edge(a, b, s=4)
        # print(a, b)

    # G2.add_edge("E4", "E6", s=3)
    # G2.add_edge("E3", "E8", s=3)
    # G2.add_edge("E7", "E14", s=3)
    # G2.add_edge("E8", "E9", s=3)
    # G2.add_edge("E12", "E13", s=3)

    # G2.add_edge("P1", "E4", s=4)
    # G2.add_edge("P1", "E8", s=4)
    # G2.add_edge("P3", "E7", s=4)
    # G2.add_edge("P4", "E14", s=4)

    if use_fixed_wt:
        mat_weight2 = get_weight_matrix_fixed(G2)
    else:
        mat_weight2 = get_weight_matrix(G2)

    return mat_weight, mat_weight1, mat_weight2


def get_weight_matrix(G):
    for edge in G.edges:
        if "weight" in G.edges[edge]:
            continue
        # CNV to gene
        if G.edges[edge]["s"] == 1:
            G.edges[edge]["weight"] = np.random.randn()*0.5
            # G.edges[edge]["weight"] = 0.5
        # CNV to gene, cis-correlation
        if G.edges[edge]["s"] == 2:
            G.edges[edge]["weight"] = np.random.randn()*0.5
            # G.edges[edge]["weight"] = 0.5
        # Gene-gene
        if G.edges[edge]["s"] == 3:
            G.edges[edge]["weight"] = np.random.randn() * 0.5
        # TF-gene
        if G.edges[edge]["s"] == 4:
            G.edges[edge]["weight"] = np.random.randn() * 0.5
    mat_weight = nx.adjacency_matrix(G).todense()
    return mat_weight


def get_weight_matrix_fixed(G):
    for edge in G.edges:
        if "weight" in G.edges[edge]:
            continue
        # CNV to gene
        if G.edges[edge]["s"] == 1:
            G.edges[edge]["weight"] = 0.5
        # CNV to gene, cis-correlation
        if G.edges[edge]["s"] == 2:
            G.edges[edge]["weight"] = 0.5
        # Gene-gene
        if G.edges[edge]["s"] == 3:
            if np.random.rand() > 0.5:
                G.edges[edge]["weight"] = 0.5
            else:
                G.edges[edge]["weight"] = 0.5
        # TF-gene
        if G.edges[edge]["s"] == 4:
            if np.random.rand() > 0.5:
                G.edges[edge]["weight"] = 0.5
            else:
                G.edges[edge]["weight"] = 0.5
    mat_weight = nx.adjacency_matrix(G).todense()
    return mat_weight


def make_dep_mask(
    n_gene=15,
    n_tf=5,
    use_cnv='diag',
    use_tf='all',
):
    n_node = 2 * n_gene + n_tf
    dep_mat = np.zeros((n_node, n_node))
    # CNV to gene is one to one
    if use_cnv == 'diag':
        dep_mat[:n_gene, n_gene : 2 * n_gene] = np.eye(n_gene)
    elif use_cnv == 'all':
        dep_mat[:n_gene, n_gene : 2 * n_gene] = 1
    else:
        dep_mat[:n_gene, n_gene : 2 * n_gene] = 0
    # gene-gene
    dep_mat[n_gene : 2 * n_gene, n_gene : 2 * n_gene] = 1
    # TF-gene
    if use_tf == 'all':
        dep_mat[2 * n_gene :, n_gene : 2 * n_gene] = 1
    return dep_mat


def make_lambda_mat(
    n_gene=15,
    n_tf=5,
    lambda1=0.3,
    lambda2=0.05,
    penalize_cnv=False,
):
    n_node = 2 * n_gene + n_tf
    lambda1_mat = np.zeros((n_node, n_node))
    lambda2_mat = np.zeros((n_node, n_node))
    # CNV to gene is one to one
    if penalize_cnv:
        lambda1_mat[:n_gene, n_gene : 2 * n_gene] = lambda1
        lambda2_mat[:n_gene, n_gene : 2 * n_gene] = lambda2
    else:
        lambda1_mat[:n_gene, n_gene : 2 * n_gene] = 0
        lambda2_mat[:n_gene, n_gene : 2 * n_gene] = 0
    # gene-gene
    lambda1_mat[n_gene : 2 * n_gene, n_gene : 2 * n_gene] = lambda1
    lambda2_mat[n_gene : 2 * n_gene, n_gene : 2 * n_gene] = lambda2
    # TF-gene
    lambda1_mat[2 * n_gene :, n_gene : 2 * n_gene] = lambda1
    lambda2_mat[2 * n_gene :, n_gene : 2 * n_gene] = lambda2
    return lambda1_mat, lambda2_mat
