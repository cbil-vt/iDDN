import pickle
import numpy as np
from iddn_extra import sim3_network_topo as nett
from iddn_extra import sim3_ode


def toy_example():
    # molecular network for TF protein and mRNA
    # Feedback not used here
    # layer. 0: RNA, 1: protein
    # role of a parent: 0: TF activate, 1: TF repress, 2: dosage in translation, 3: miRNA repress

    idx_layer = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
    idx_par = {
        0: [],
        1: [5],
        2: [6],
        3: [6, 7],
        4: [7],
        5: [],
        6: [],
        7: [0, 1],
        8: [2],
        9: [3, 4],
    }
    idx_par_roles = {
        0: [],
        1: [0],
        2: [0],
        3: [0, 1],
        4: [1],
        5: [],
        6: [],
        7: [3, 2],
        8: [2],
        9: [3, 2],
    }
    _, state_history = sim3_ode.run_sim(
        idx_layer, idx_par, idx_par_roles, n_sample=1, n_max_steps=1000
    )
    return idx_layer, idx_par, idx_par_roles, state_history


def sim_pipeline(
    idx_rep=0,
    n_top_regulator=3,
    n_second_regulator=27,
    n_other_genes=70,
    wt_top_regulator=3,
    two_condition_ratio=0.25,
    n_sample=1000,
    n_steps=1000,
    dt=0.02,
    method="ode",
    out_folder="./sim_input/",
):
    # Make a gene network
    gene_type, gene_par = nett.create_gene_network(
        n_top_regulator=n_top_regulator,
        n_second_regulator=n_second_regulator,
        n_other_genes=n_other_genes,
        wt_top_regulator=wt_top_regulator,
    )
    n_genes = len(gene_type)

    # Make a molecule level network
    mol_layer, mol_par, mol_par_roles, mol_type = nett.create_mol_network(
        gene_type,
        gene_par,
    )

    # Make two conditions by removing some edges in each condition
    mol_par1, mol_par2, mol_par_roles1, mol_par_roles2 = (
        nett.make_two_conditions_mol_net(
            mol_par,
            mol_par_roles,
            ratio=two_condition_ratio,
        )
    )

    # Simulation for each condition
    net_info1, dep_mat1, con_mat1 = nett.prep_net_for_sim(
        mol_layer, mol_par1, mol_par_roles1, mol_type
    )
    net_info2, dep_mat2, con_mat2 = nett.prep_net_for_sim(
        mol_layer, mol_par2, mol_par_roles2, mol_type
    )

    dat_sim1, state_history1 = sim3_ode.run_sim(
        net_info1["idx_layer"],
        net_info1["idx_par"],
        net_info1["idx_par_roles"],
        n_sample=n_sample,
        n_max_steps=n_steps,
        dt=dt,
        method=method,
    )
    dat_sim2, state_history2 = sim3_ode.run_sim(
        net_info2["idx_layer"],
        net_info2["idx_par"],
        net_info2["idx_par_roles"],
        n_sample=n_sample,
        n_max_steps=n_steps,
        dt=dt,
        method=method,
    )

    # Save results and networks
    pickle.dump(
        [net_info1, net_info2],
        open(
            f"{out_folder}/gene_{n_genes}_{method}_rep{idx_rep}_sample_{n_sample}.pkl",
            "wb",
        ),
    )
    np.savez(
        f"{out_folder}/gene_{n_genes}_{method}_rep{idx_rep}_sample_{n_sample}.npz",
        dat_sim1=dat_sim1,
        state_history1=state_history1,
        dep_mat1=dep_mat1,
        con_mat1=con_mat1,
        dat_sim2=dat_sim2,
        state_history2=state_history2,
        dep_mat2=dep_mat2,
        con_mat2=con_mat2,
    )

    out = dict(
        net1=net_info1,
        dat1=dat_sim1,
        state_history1=state_history1,
        dep_mat1=dep_mat1,
        con_mat1=con_mat1,
        net2=net_info2,
        dat2=dat_sim2,
        state_history2=state_history2,
        dep_mat2=dep_mat2,
        con_mat2=con_mat2,
    )

    return out
