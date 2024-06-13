import numpy as np
import networkx as nx
from iddn import iddn
from ddn3 import performance


def barabasi_albert_digraph(n, m_min, m_max, n_input, rep_init=5):
    G = nx.DiGraph()

    # nodes with external inputs
    G.add_nodes_from(range(n_input))

    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = list(range(n_input)) * rep_init

    # Start adding the other n - m0 nodes.
    source = len(G)
    while source < n:
        # Choose m0 unique nodes from the existing nodes
        # Each time we add from 1 to m edges
        # Pick uniformly from repeated_nodes (preferential attachment)
        m0 = np.random.randint(m_min, m_max + 1)
        if len(repeated_nodes) > m0:
            targets = np.random.choice(repeated_nodes, m0, replace=False)
        else:
            targets = np.random.choice(repeated_nodes, 1, replace=False)

        # Add edges to m nodes from the source.
        G.add_edges_from(zip(targets, [source] * m0))

        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)

        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m0)

        source += 1
    return G


def prep_sim_from_graph(G):
    regu_edges = {}
    for key, val in dict(G.in_degree).items():
        if val > 0:
            edges_in = list(G.in_edges(key))
            nodes_in = np.sort(np.array([x[0] for x in edges_in]))
            if len(nodes_in) == 2:
                if np.random.rand() > 0.5:
                    signs_in = np.array([1, 0])
                else:
                    signs_in = np.array([0, 1])
            else:
                signs_in = np.random.randint(0, 2, len(nodes_in))
            regu_edges[key] = (nodes_in, signs_in)
        else:
            regu_edges[key] = ((), ())

    return regu_edges


def graph_bb_to_whole(G_bb: nx.Graph, n_input, n_lnc, n_lnc_out=2, weight_cog=0.8):
    n_genes = len(G_bb)

    regu_edges = prep_sim_from_graph(G_bb)

    out_deg = np.array(G_bb.out_degree)[:, 1]
    print("Leaf nodes number: ", np.sum(out_deg == 0))
    # tf_nodes = np.where(out_deg>0)[0]

    # genes and proteins
    G = nx.DiGraph()
    for key, edges in regu_edges.items():
        node_gene = f"mrna_{key:03d}"
        node_prot = f"prot_{key:03d}"
        # weight_cog = 0.8
        # weight_cog = 0.95
        G.add_edge(node_gene, node_prot, weight=weight_cog)
        for i in range(len(edges[0])):
            node_regu = f"prot_{edges[0][i]:03d}"
            G.add_edge(node_regu, node_gene, weight=edges[1][i] - 0.5)

    # long non-coding RNA
    # n_lnc = 40
    # n_lnc_in = 1
    # n_lnc_out = 2

    genes_regulated_by_lnc = np.arange(n_input, n_genes)

    for i in range(n_lnc):
        # idx_in = np.random.choice(tf_nodes, n_lnc_in)
        idx_out = np.random.choice(genes_regulated_by_lnc, n_lnc_out)
        # wt = np.random.randint(low=0, high=2, size=n_lnc_in) - 0.5
        # for idx, j in enumerate(idx_in):
        #     G.add_edge(f"prot_{j:03d}", f"rlnc_{i:03d}", weight=wt[idx])
        wt = np.random.randint(low=0, high=2, size=n_lnc_out) - 0.5
        for idx, j in enumerate(idx_out):
            G.add_edge(f"rlnc_{i:03d}", f"mrna_{j:03d}", weight=wt[idx])

    return G


def graph_to_matrix(G: nx.DiGraph):
    node_names = list(G.nodes())
    node_names.sort()

    # idx_node_dict = {idx:name for idx, name in enumerate(node_names)}
    node_idx_dict = {name: idx for idx, name in enumerate(node_names)}

    # weight matrix
    n_node = len(G)

    mat_wt = np.zeros((n_node, n_node))
    for edge in G.edges():
        wt = G.edges[edge]["weight"]
        n0 = node_idx_dict[edge[0]]
        n1 = node_idx_dict[edge[1]]
        mat_wt[n0, n1] = wt

    # mat_wt_sym = mat_wt + mat_wt.T
    gt_net = 1 * (mat_wt != 0)
    # gt_net_sym = 1*(mat_wt_sym!=0)

    idx_mrna = get_node_subset(node_idx_dict, "mrna")
    idx_prot = get_node_subset(node_idx_dict, "prot")

    # ground truth between protein and mRNA
    gt_sub = gt_net[np.ix_(idx_prot, idx_mrna)]

    # ground truth for gene-gene interaction
    gt_sub_sym = gt_sub + gt_sub.T

    return node_idx_dict, mat_wt, gt_sub_sym


def sim_steady_state_linear(
    mat_wt,
    n_sample,
):
    # for each node, find inputs and weights
    n_node = len(mat_wt)
    node_in_lst = []
    noise_scl_in_lst = []
    dat_scl_in_lst = []
    for i in range(n_node):
        idx = np.where(mat_wt[:, i] != 0)[0]
        wt = mat_wt[:, i][idx]
        sign = np.ones_like(wt)
        sign[wt < 0] = -1
        scl = np.sqrt((1 / wt) ** 2 - 1)
        node_in_lst.append(idx)
        noise_scl_in_lst.append(scl)
        dat_scl_in_lst.append(sign)

    # run simulation
    dat = np.zeros((n_node, n_sample))
    stable_mask = np.zeros(n_node)
    n_max_steps = n_node

    for i in range(n_max_steps):
        node_lst = np.random.permutation(n_node)
        if int(np.sum(stable_mask)) == n_node:
            break

        for node_cur in node_lst:
            if stable_mask[node_cur] == 1:
                continue

            if len(node_in_lst[node_cur]) == 0:
                # if this node has no input, use N(0,1)
                # then set it as stable and no longer update
                x = np.random.randn(n_sample)
                stable_mask[node_cur] = 1
            else:
                # if all inputs are already stable, set current node as stable
                if np.sum(stable_mask[node_in_lst[node_cur]] == 0) == 0:
                    stable_mask[node_cur] = 1
                x = np.zeros(n_sample)

                # include the contribution of each input
                for i, node in enumerate(node_in_lst[node_cur]):
                    wt_dat = dat_scl_in_lst[node_cur][i]
                    wt_noise = noise_scl_in_lst[node_cur][i]
                    x += dat[node] * wt_dat + np.random.randn(n_sample) * wt_noise

                # scale to N(0,1)
                if np.sum(noise_scl_in_lst[node_cur]) > 0:
                    x = x - np.mean(x)
                    x = x / np.std(x)
            dat[node_cur] = x
    print(np.sum(stable_mask))
    dat = dat.T

    return dat


def run_ddn(
    dat,
    node_idx_dict,
    gt_sub_sym,
    rho1_lst,
):
    n_node = dat.shape[1]
    idx_mrna = get_node_subset(node_idx_dict, "mrna")

    # dependency for DDN, use mRNA only
    dep_mat_ddn = np.zeros((n_node, n_node))
    dep_mat_ddn[np.ix_(idx_mrna, idx_mrna)] = 1

    rho1_mat_ddn = np.copy(dep_mat_ddn)
    rho2_mat_ddn = np.copy(dep_mat_ddn)

    res_ddn = np.zeros((len(rho1_lst), 5))
    gg_lst_ddn = []
    for i, lambda1 in enumerate(rho1_lst):
        lambda1_mat = rho1_mat_ddn * lambda1
        lambda2_mat = rho2_mat_ddn
        out_ddn = iddn.iddn(
            dat,
            dat,
            lambda1=lambda1_mat,
            lambda2=lambda2_mat,
            dep_mat=dep_mat_ddn,
        )
        gg_est = out_ddn[0][np.ix_(idx_mrna, idx_mrna)]
        gg_est = sim_tools.clean_adjacency(gg_est)
        gg_lst_ddn.append(gg_est)
        res_ddn[i] = performance.get_error_measure_two_theta(gg_est, gt_sub_sym)

    return res_ddn, dep_mat_ddn


def run_iddn(
    dat,
    node_idx_dict,
    gt_sub_sym,
    rho1_lst,
    use_lnc=True,
):
    n_node = dat.shape[1]

    idx_mrna = get_node_subset(node_idx_dict, "mrna")
    idx_prot = get_node_subset(node_idx_dict, "prot")
    idx_rlnc = get_node_subset(node_idx_dict, "rlnc")

    # dependency for realistic simulation for iDDN
    dep_mat_iddn = np.zeros((n_node, n_node))

    dep_mat_iddn[np.ix_(idx_prot, idx_mrna)] = 1
    if use_lnc:
        dep_mat_iddn[np.ix_(idx_rlnc, idx_mrna)] = 1

    dep_mat_iddn[idx_mrna, idx_prot] = 0
    dep_mat_iddn[idx_prot, idx_mrna] = 0

    rho1_mat_iddn = np.copy(dep_mat_iddn)
    # rho1_mat_iddn[idx_mrna, idx_prot] = 0
    # rho1_mat_iddn[idx_prot, idx_mrna] = 0
    rho2_mat_iddn = np.copy(dep_mat_iddn)

    res_iddn = np.zeros((len(rho1_lst), 5))
    gg_lst_iddn = []
    for i, lambda1 in enumerate(rho1_lst):
        lambda1_mat = rho1_mat_iddn * lambda1
        lambda2_mat = rho2_mat_iddn
        out_iddn = iddn.iddn(
            dat,
            dat,
            lambda1=lambda1_mat,
            lambda2=lambda2_mat,
            dep_mat=dep_mat_iddn,
        )
        gg_est = out_iddn[0][np.ix_(idx_prot, idx_mrna)]
        gg_est = sim_tools.clean_adjacency(gg_est)
        gg_lst_iddn.append(gg_est)
        res_iddn[i] = performance.get_error_measure_two_theta(gg_est, gt_sub_sym)

    return res_iddn, dep_mat_iddn


def get_node_subset(node_idx_dict: dict, query: str = "mrna"):
    n = len(query)
    return [idx for node, idx in node_idx_dict.items() if node[:n] == query]


################################################################
# SynTReN like
################################################################


def sim_syntren_like(
    regu_edges,
    gene_val_lst,
    mrna_post_scale_lst,
    n_steps=10,
    noise_sigma=0.1,
    hill_coef=4,
    hill_thr=1,
):
    n_genes = len(gene_val_lst)

    # make sure the dynamic range from 0 to 2, even there is no repressor or activator
    node_regu_ofst = np.zeros(n_genes)
    node_regu_scale = np.ones(n_genes)

    for key, val in regu_edges.items():
        # n_act = np.sum(val[1] == 1)
        # n_rep = np.sum(val[1] == 0)
        # node_regu_scale[key] = 1/(0.5**n_rep * (1.5)**n_act)

        if np.sum(val[1] == 1) == 0:
            # all repressor
            node_regu_scale[key] = 2
        if np.sum(val[1] == 0) == 0:
            # all activator
            node_regu_ofst[key] = -1
            node_regu_scale[key] = 2

    for _ in range(n_steps):
        # protein production with randomness
        prot_temp = gene_val_lst * mrna_post_scale_lst
        noise = np.random.randn(n_genes) * noise_sigma  # FIXME: more realistic noise
        prot_val_noisy = prot_temp + noise
        prot_Val_hill = (prot_val_noisy / hill_thr) ** hill_coef

        # activator
        # NOTE: maybe use a smaller offset is there are multiple activators
        prot_act = 1 + prot_Val_hill / (1 + prot_Val_hill)
        # repressor
        prot_rep = 1 / (1 + prot_Val_hill)

        # regulation of mRNA levels
        for gene_idx, edge in regu_edges.items():
            n_regulators = len(edge[0])
            mrna_val = 1
            # if gene_idx==9:
            #     print('hi')
            for i in range(n_regulators):
                tf_idx = edge[0][i]
                sign = edge[1][i]
                if sign > 0:
                    tf_indi = prot_act[tf_idx]
                else:
                    tf_indi = prot_rep[tf_idx]
                mrna_val = mrna_val * tf_indi

            mrna_val = (mrna_val + node_regu_ofst[gene_idx]) * node_regu_scale[gene_idx]
            gene_val_lst[gene_idx] = mrna_val + np.random.randn() * 0.1 * mrna_val

    return gene_val_lst, prot_val_noisy
