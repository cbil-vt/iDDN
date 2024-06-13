import numpy as np


def create_gene_network(
    n_top_regulator=3,
    n_second_regulator=7,
    n_other_genes=10,
    wt_top_regulator=3,
):
    # We begin with a gene network, then translate to molecular level
    # A top regulator is treated as external input
    # Second regulators are regulated by other regulators, include top and second ones
    # Other genes are protein coding genes that are not TFs

    n_regulators = n_top_regulator + n_second_regulator
    regulator_idx = np.arange(n_regulators)

    regulator_weight = np.ones(n_regulators)
    regulator_weight[:n_top_regulator] = wt_top_regulator

    n_gene = n_regulators + n_other_genes

    gene_type = dict()
    gene_parent = dict()
    for i in range(n_gene):
        gene_parent[i] = []

        if i < n_top_regulator:
            gene_type[i] = "top_reg"
        elif i < n_regulators:
            gene_type[i] = "reg"
        else:
            gene_type[i] = "other"

        if gene_type[i] == "reg":
            parents = regulator_idx[:i]
            parents_wt = regulator_weight[:i]
            parents_wt = parents_wt / np.sum(parents_wt)
            thr2 = 0.2
        elif gene_type[i] == "other":
            parents = regulator_idx[:n_regulators]
            parents_wt = regulator_weight[:n_regulators]
            parents_wt = parents_wt / np.sum(parents_wt)
            thr2 = 0.5
        else:
            continue

        # choose one or two regulators
        p = np.random.choice(parents, size=2, replace=False, p=parents_wt).tolist()
        if np.random.rand() < thr2:
            gene_parent[i].extend(p)
        else:
            gene_parent[i].append(p[0])

    return gene_type, gene_parent


def create_mol_network(gene_type, gene_parent):
    # molecular network for TF protein and mRNA
    # Feedback not used here
    # molecule species (level in the ODE). 0: RNA, 1: protein
    # molecule type: mRNA, miRNA, TFmRNA, lncRNA, protein, etc.
    # role of a parent: 0: TF activate, 1: TF repress, 2: dosage in translation, 3: miRNA repress

    # mol_layer = {0:0 ,1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1, 8:1, 9:1}
    # mol_parent = {0:[], 1:[5], 2:[6], 3:[6, 7], 4:[7], 5:[], 6:[], 7:[0, 1], 8:[2], 9:[3,4]}
    # mol_parent_roles = {0:[], 1:[0], 2:[0], 3:[0, 1], 4:[1], 5:[], 6:[], 7:[3, 2], 8:[2], 9:[3,2]}

    # All nodes
    mol_type = dict()  # not needed for final ODE, only used for building network
    mol_layer = dict()
    n_gene = len(gene_type)
    for i in range(n_gene):
        if gene_type[i] == "top_reg" or gene_type[i] == "reg":
            if np.random.rand() < 0.3:
                # Assign some regulators as miRNAs
                mol_layer[f"rna_{i}"] = 0
                mol_type[f"rna_{i}"] = "miRNA"
            else:
                mol_layer[f"rna_{i}"] = 0
                mol_layer[f"prot_{i}"] = 1
                mol_type[f"rna_{i}"] = "TFmRNA"
                mol_type[f"prot_{i}"] = "protein"
        else:
            mol_layer[f"rna_{i}"] = 0
            mol_layer[f"prot_{i}"] = 1
            mol_type[f"rna_{i}"] = "mRNA"
            mol_type[f"prot_{i}"] = "protein"

    # All edges
    mol_parent = dict()
    mol_parent_roles = dict()
    for x in mol_layer.keys():
        mol_parent[x] = []
        mol_parent_roles[x] = []

    for x, px_lst in gene_parent.items():
        # RNA to protein if x is a protein coding gene
        if f"prot_{x}" in mol_layer:
            mol_parent[f"prot_{x}"].append(f"rna_{x}")
            mol_parent_roles[f"prot_{x}"].append(2)
        # edges from regulator to RNA of x
        for px in px_lst:
            if mol_type[f"rna_{px}"] == "miRNA":
                if f"prot_{x}" in mol_layer:
                    # parent is miRNA. We assume miRNA only repress the protein expression
                    mol_parent[f"prot_{x}"].append(f"rna_{px}")
                    mol_parent_roles[f"prot_{x}"].append(3)
            else:
                # parent is TF
                mol_parent[f"rna_{x}"].append(f"prot_{px}")
                mol_parent_roles[f"rna_{x}"].append(np.random.randint(2))

    return mol_layer, mol_parent, mol_parent_roles, mol_type


def make_two_conditions_mol_net(mol_parent, mol_parent_roles, ratio=0.25):
    # Node is not changed, just remove edges
    # Assume the ratio parameter is 25%
    # For each regulatory edge, 25% chance to condition 1 only, 25% chance to condition 2 only
    # 50% chance to both.

    mol_parent1 = dict()
    mol_parent2 = dict()
    mol_parent_roles1 = dict()
    mol_parent_roles2 = dict()

    for mol_name in mol_parent.keys():
        par_now = mol_parent[mol_name]
        par_role_now = mol_parent_roles[mol_name]

        mol_parent1[mol_name] = []
        mol_parent2[mol_name] = []
        mol_parent_roles1[mol_name] = []
        mol_parent_roles2[mol_name] = []

        for i in range(len(par_now)):
            xx = np.random.rand()

            if (mol_parent_roles[mol_name][i] == 2) or (xx > 2 * ratio):
                # ignore the translation edge
                mol_parent1[mol_name].append(par_now[i])
                mol_parent2[mol_name].append(par_now[i])
                mol_parent_roles1[mol_name].append(par_role_now[i])
                mol_parent_roles2[mol_name].append(par_role_now[i])
                continue

            if xx < ratio:
                # assign to condition 1
                mol_parent1[mol_name].append(par_now[i])
                mol_parent_roles1[mol_name].append(par_role_now[i])
            elif xx < 2 * ratio:
                # assign to condition 2
                mol_parent2[mol_name].append(par_now[i])
                mol_parent_roles2[mol_name].append(par_role_now[i])
            else:
                raise "Something wrong"

    return mol_parent1, mol_parent2, mol_parent_roles1, mol_parent_roles2


def mol_network_to_index(mol_layer, mol_parent, mol_parent_roles):
    # change molecular names to indices
    mol_to_idx = dict()
    idx_to_mol = dict()
    i = 0
    for x in mol_layer.keys():
        mol_to_idx[x] = i
        idx_to_mol[i] = x
        i = i + 1

    idx_layer = dict()
    idx_parent = dict()
    idx_parent_roles = dict()
    for x, px in mol_layer.items():
        idx_layer[mol_to_idx[x]] = px

    for x, px in mol_parent.items():
        idx_parent[mol_to_idx[x]] = [mol_to_idx[px0] for px0 in px]

    for x, px in mol_parent_roles.items():
        idx_parent_roles[mol_to_idx[x]] = px

    return idx_layer, idx_parent, idx_parent_roles, mol_to_idx, idx_to_mol


def get_dep_mat(idx_parent):
    n_node = len(idx_parent)
    dep_mat = np.zeros((n_node, n_node))
    for x, px in idx_parent.items():
        dep_mat[px, x] = 1

    con_mat = dep_mat + dep_mat.T
    con_mat[con_mat > 1] = 1
    return dep_mat, con_mat


def get_translation_mat(idx_parent, idx_parent_roles):
    n_node = len(idx_parent)
    translation_mat = np.zeros((n_node, n_node))
    for x in range(n_node):
        px = idx_parent[x]
        px_roles = idx_parent_roles[x]
        for i in range(len(px)):
            if px_roles[i] == 2:
                translation_mat[px[i], x] = 1
                translation_mat[x, px[i]] = 1

    return translation_mat


def prep_net_for_sim(mol_layer, mol_par, mol_par_roles, mol_type):
    idx_layer, idx_par, idx_par_roles, mol2idx, idx2mol = mol_network_to_index(
        mol_layer, mol_par, mol_par_roles
    )
    net_info = dict(
        mol_layer=mol_layer,
        mol_par=mol_par,
        mol_par_roles=mol_par_roles,
        mol_type=mol_type,
        idx_layer=idx_layer,
        idx_par=idx_par,
        idx_par_roles=idx_par_roles,
        mol2idx=mol2idx,
        idx2mol=idx2mol,
    )
    dep_mat, con_mat = get_dep_mat(idx_par)
    return net_info, dep_mat, con_mat

########################################################################################################################
# For toy simulation
########################################################################################################################

def _add_one_gene(mol_layer, mol_par, mol_par_roles, gene_name, par_names):
    mrna_name = gene_name + "_mrna"
    prot_name = gene_name + "_prot"
    mol_layer[mrna_name] = 0
    mol_layer[prot_name] = 1
    mol_par[mrna_name] = par_names
    mol_par[prot_name] = [mrna_name]
    if len(par_names) > 0:
        mol_par_roles[mrna_name] = np.random.randint(low=0, high=2, size=len(par_names)).tolist()
    else:
        mol_par_roles[mrna_name] = []
    mol_par_roles[prot_name] = [2]


def add_one_hub_net(mol_layer, mol_par, mol_par_roles, hub_to_tf, hub_to_gene, tf_to_gene, net_idx):
    gene_name = f"hub_{net_idx}_tf_hub"
    _add_one_gene(mol_layer, mol_par, mol_par_roles, gene_name, [])

    for i in range(hub_to_gene):
        gene_name = f"hub_{net_idx}_tf_hub_gene_{i}"
        _add_one_gene(mol_layer, mol_par, mol_par_roles, gene_name, [f"hub_{net_idx}_tf_hub_prot"])

    for i in range(hub_to_tf):
        gene_name = f"hub_{net_idx}_tf_{i}"
        _add_one_gene(mol_layer, mol_par, mol_par_roles, gene_name, [f"hub_{net_idx}_tf_hub_prot"])

    for i in range(hub_to_tf):
        for j in range(tf_to_gene):
            gene_name = f"hub_{net_idx}_tf_{i}_gene_{j}"
            _add_one_gene(mol_layer, mol_par, mol_par_roles, gene_name, [f"hub_{net_idx}_tf_{i}_prot"])


def molnet_to_genenet(mol_par):
    # make gene network from molecular network
    # use the names of molecules to group them
    gene_mol = dict()
    for x in mol_par.keys():
        x_base = x[:-5]
        if not x_base in gene_mol:
            gene_mol[x_base] = [x]
        else:
            gene_mol[x_base].append(x)

    gene_par = dict()
    for x, px in mol_par.items():
        x_base = x[:-5]
        if not x_base in gene_par:
            gene_par[x_base] = []
        for px0 in px:
            px0_base = px0[:-5]
            if px0_base not in gene_par[x_base]:
                if px0_base != x_base:
                    gene_par[x_base].append(px0_base)

    gene2idx = dict()
    idx2gene = dict()
    i = 0
    for x in gene_par:
        gene2idx[x] = i
        idx2gene[i] = x
        i += 1

    n_genes = len(gene_par)
    gene_dep_mat = np.zeros((n_genes, n_genes))
    for x, px in gene_par.items():
        for px0 in px:
            x_idx = gene2idx[x]
            px0_idx = gene2idx[px0]
            gene_dep_mat[px0_idx, x_idx] = 1
    gene_con_mat = 1*(gene_dep_mat + gene_dep_mat.T)>0

    return gene_par, gene_mol, gene2idx, idx2gene, gene_dep_mat, gene_con_mat


def molcon_to_genecon(mol_con_est, idx2mol, gene2idx, thr=1e-4):
    # n_mol = len(idx2mol)
    # mol_con_est = 1 * (np.random.rand(n_mol, n_mol) > 0.99)

    n_gene = len(gene2idx)
    gene_con_est = np.zeros((n_gene, n_gene))
    m0_idx_lst, m1_idx_lst = np.where(np.abs(mol_con_est) > thr)
    for i in range(len(m0_idx_lst)):
        m0_idx = m0_idx_lst[i]
        m1_idx = m1_idx_lst[i]
        m0 = idx2mol[m0_idx]
        m1 = idx2mol[m1_idx]
        g0 = m0[:-5]
        g1 = m1[:-5]
        g0_idx = gene2idx[g0]
        g1_idx = gene2idx[g1]
        gene_con_est[g0_idx, g1_idx] = 1
        gene_con_est[g1_idx, g0_idx] = 1

    return gene_con_est


def make_iddn_dep_prior(mol_layer, allow_mrna_pairs=False, allow_protein_pairs=False):
    # iDDN dependency constraints
    n_mol = len(mol_layer)
    dep_mat_prior = np.zeros((n_mol, n_mol))
    mol_names = list(mol_layer.keys())
    mol_type = []
    for m in mol_names:
        if "gene" in m:
            if m.endswith("_mrna"):
                mol_type.append("gene mrna")
            else:
                mol_type.append("gene prot")
        else:
            if m.endswith("_mrna"):
                mol_type.append("tf mrna")
            else:
                mol_type.append("tf prot")

    for i0, m0 in enumerate(mol_names):
        for i1, m1 in enumerate(mol_names):
            # cognate pairs
            if m0[:-5] == m1[:-5]:
                if m0.endswith("_mrna"):
                    dep_mat_prior[i0, i1] = 1
                else:
                    dep_mat_prior[i1, i0] = 1

            if mol_type[i0] == "gene mrna" and mol_type[i1] == "tf prot":
                dep_mat_prior[i1, i0] = 1
            if mol_type[i0] == "tf mrna" and mol_type[i1] == "tf prot":
                dep_mat_prior[i1, i0] = 1
            if allow_mrna_pairs:
                if mol_type[i0] == "gene mrna" and mol_type[i1] == "tf mrna":
                    dep_mat_prior[i1, i0] = 1
            if allow_protein_pairs:
                if mol_type[i0] == "gene prot" and mol_type[i1] == "tf prot":
                    dep_mat_prior[i1, i0] = 1

    return dep_mat_prior


def molmat_to_genemat(t1_lst, t2_lst, idx2mol, gene2idx):
    t1_lst_gene = []
    t2_lst_gene = []
    for t1 in t1_lst:
        t1_gene = molcon_to_genecon(t1, idx2mol, gene2idx)
        t1_lst_gene.append(t1_gene)
    for t2 in t2_lst:
        t2_gene = molcon_to_genecon(t2, idx2mol, gene2idx)
        t2_lst_gene.append(t2_gene)
    return t1_lst_gene, t2_lst_gene
