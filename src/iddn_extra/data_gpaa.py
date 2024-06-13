import numpy as np
import pandas as pd
from iddn import iddn
from ddn3 import tools_export


def load_protein(
    dat_folder="../../../x_data/gpaa/multiomics_2024/LAD_404_Samples_Combined_Omics/",
):
    # if correct_cam_prop:
    file_omics = dat_folder + "LAD_combined_omics_log2_transformed_ver2.csv"
    df_data_all = pd.read_csv(file_omics)
    df_prot = df_data_all[(df_data_all["HasProtein"] > 0)]
    return df_prot


def get_feat_names(df_lf, fea_type="TFs", fea_pre="TFmRNA"):
    fea = df_lf[fea_type].to_list()
    fea = [x for x in fea if x == x]
    fea.sort()
    fea_pre = [fea_pre + "|" + x for x in fea]
    return fea_pre, fea


def sort_nodes(nodes_show):
    node0 = list()
    node1 = list()
    node2 = list()
    for n in nodes_show:
        if n[-2:] == "_1":
            node1.append(n)
        elif n[-2:] == "_2":
            node2.append(n)
        else:
            node0.append(n)
    node0.sort()
    node0 = node0[-2:] + node0[:-2]
    node1.sort()
    node2.sort()
    nodes_show = node0 + node1 + node2
    return nodes_show


def iddn_dep_rho_mat_tf_protein(
    node_names_dict,
    rho1,
    rho2,
    tf_tf=1,
    tf_prot=1,
    use_tf_tf=True,
):
    tf = node_names_dict["tf"]
    prot = node_names_dict["prot"]

    n_tf = len(tf)
    n_prot = len(prot)
    n_node = n_tf + n_prot

    dep_mat = np.zeros((n_node, n_node))
    dep_mat[:n_tf, n_tf : n_tf + n_prot] = 1
    if use_tf_tf:
        dep_mat[:n_tf, :n_tf] = 1

    rho1_mat = np.zeros((n_node, n_node))

    # tf to protein
    rho1_mat[:n_tf, n_tf : n_tf + n_prot] = tf_prot

    # within tf
    if use_tf_tf:
        rho1_mat[:n_tf, :n_tf] = tf_tf

    rho2_mat = dep_mat

    return dep_mat, rho1_mat * rho1, rho2_mat * rho2


def run_iddn(dat0, dat1, dep_mat, lambda1_mat, lambda2_mat, node_names, n_process=1):
    out_ddn = iddn.iddn_parallel(
        dat0,
        dat1,
        lambda1=lambda1_mat,
        lambda2=lambda2_mat,
        dep_mat=dep_mat,
        n_process=n_process,
    )

    omega1 = out_ddn[0]
    omega2 = out_ddn[1]

    (
        comm_edge,
        _,
        _,
        diff_edge,
        nodes_show,
    ) = tools_export.get_diff_comm_net_for_plot(omega1, omega2, node_names)

    # only interested in common edge nodes
    s1 = set(comm_edge["gene1"].to_list())
    s2 = set(comm_edge["gene2"].to_list())
    nodes_show = s1.union(s2)
    nodes_show = sort_nodes(nodes_show)

    return comm_edge, diff_edge, nodes_show


# def sort_nodes(nodes_show):
#     from itertools import chain
#
#     # sort the nodes to show
#     xx_lst = []
#     for i in [0, 3, 1, 2]:
#         xx = []
#         for n in nodes_show:
#             if n[-1] == str(i):
#                 xx.append(n)
#         xx.sort()
#         xx_lst.append(xx)
#     nodes_show = list(chain(*xx_lst))
#     return nodes_show


def get_node_type_and_label_multi_parts_sort(
    nodes_show,
    part_id_lst=("_0", "_1", "_2", "_3"),
):
    nodes_show = sort_nodes(nodes_show)

    x_len_lst = [len(x) for x in part_id_lst]
    nodes_type = dict()
    labels = dict()
    for i, node in enumerate(nodes_show):
        for j in range(len(x_len_lst)):
            if node[-x_len_lst[j] :] == part_id_lst[j]:
                labels[node] = node[: -x_len_lst[j]]
                if node[-x_len_lst[j] :] != "_3":
                    nodes_type[node] = (j, j)
                else:
                    nodes_type[node] = (0, j)

    return nodes_show, nodes_type, labels
