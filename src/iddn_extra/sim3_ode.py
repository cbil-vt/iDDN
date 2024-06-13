import numpy as np
from tqdm import tqdm
import numba
from math import sqrt


# @numba.njit
def calculate_promoter_activity(
    state_prev, parents, parents_roles, n_parent, hill_thr, hill_coef
):
    # the expression probability
    f = 1

    if n_parent == 1:
        p = (state_prev[parents[0]] / hill_thr) ** hill_coef
        if parents_roles[0] == 0:
            f = p / (1 + p)
        else:
            f = 1 / (1 + p)
    elif n_parent == 2:
        p0 = (state_prev[parents[0]] / hill_thr) ** hill_coef
        p1 = (state_prev[parents[1]] / hill_thr) ** hill_coef
        pa0_role = parents_roles[0]
        pa1_role = parents_roles[1]
        # only model the AND logic for two regulators
        if pa0_role == 0 and pa1_role == 0:
            f = p0 * p1 / (1 + p0 + p1 + p0 * p1)
        if pa0_role == 0 and pa1_role == 1:
            f = p0 / (1 + p0 + p1 + p0 * p1)
        if pa0_role == 1 and pa1_role == 0:
            f = p1 / (1 + p0 + p1 + p0 * p1)
        if pa0_role == 1 and pa1_role == 1:
            f = 1 / (1 + p0 + p1 + p0 * p1)
    else:
        # only simulate 0 to 2 inputs
        raise ValueError

    return f


# @numba.njit
def calculate_translation_activity(
    state_prev, parents, parents_roles, n_parent, hill_thr, hill_coef
):
    # group parents to mRNA and miRNA
    idx_mrna = -1
    idx_mirna = []
    for i in range(n_parent):
        if parents_roles[i] == 2:
            idx_mrna = parents[i]
        if parents_roles[i] == 3:
            idx_mirna.append(parents[i])

    # regulation strength of miRNA
    if len(idx_mirna) == 0:
        f = 0
    elif len(idx_mirna) == 1:
        p = (state_prev[idx_mirna[0]] / hill_thr) ** hill_coef
        f = p / (1 + p)
    elif len(idx_mirna) == 2:
        # use OR logic for miRNA
        p0 = (state_prev[idx_mirna[0]] / hill_thr) ** hill_coef
        p1 = (state_prev[idx_mirna[1]] / hill_thr) ** hill_coef
        f = (p0 + p1 + p0 * p1) / (1 + p0 + p1 + p0 * p1)
    else:
        # only simulate 0 to 2 inputs
        # TODO: support 3 or 4 miRNAs
        raise ValueError

    return f, idx_mrna


# @numba.njit
def ode_one_step(
    state_history,
    cur_step,
    idx_layer,
    idx_parent,
    idx_parent_roles,
    n_parent,
    hill_thr=10,
    hill_coef=4,
    transcription_rate=20,
    translation_rate=1,
    mrna_degradation_rate=1,
    protein_degradation_rate=1,
    dt=1e-2,
):
    n_node = len(idx_layer)
    prev_step = cur_step - 1
    state_prev = state_history[prev_step]
    for idx in range(n_node):
        species = idx_layer[idx]
        parents = idx_parent[idx]
        parents_roles = idx_parent_roles[idx]
        n_parent_idx = n_parent[idx]

        # if not regulated, consider as external input
        if n_parent_idx == 0:
            state_history[cur_step][idx] = state_prev[idx]
            continue

        if species == 0:  # transcription, TF -> mRNA
            f = calculate_promoter_activity(
                state_prev, parents, parents_roles, n_parent_idx, hill_thr, hill_coef
            )
            dx = (f * transcription_rate - mrna_degradation_rate * state_prev[idx]) * dt
            state_history[cur_step][idx] = state_prev[idx] + dx
        elif species == 1:  # translation, mRNA + miRNA -> protein
            mirna_activity, idx_mrna = calculate_translation_activity(
                state_prev, parents, parents_roles, n_parent_idx, hill_thr, hill_coef
            )
            # miRNA reduce the translation rate
            dp = (
                (translation_rate - mirna_activity) * state_prev[idx_mrna]
                - protein_degradation_rate * state_prev[idx]
            ) * dt
            state_history[cur_step][idx] = state_prev[idx] + dp
        else:
            raise ValueError


def steady_state_one_step(
    state_history,
    noise_history,
    node_ready_time,
    cur_step,
    idx_layer,
    idx_parent,
    idx_parent_roles,
    n_parent,
    sigma=0.2,
    hill_thr=10,
    hill_coef=1,  # 4
    transcription_rate=20,
    translation_rate=1,
    mrna_degradation_rate=1,
    protein_degradation_rate=1,
    # shrink_ratio=4,  # 1
):
    # SynTReN like steady state simulation on DAG
    n_node = len(idx_layer)
    prev_step = cur_step - 1
    state_prev = state_history[prev_step]
    for idx in range(n_node):
        species = idx_layer[idx]
        parents = idx_parent[idx]
        parents_roles = idx_parent_roles[idx]
        n_parent_idx = n_parent[idx]
        # print(idx, parents, parents_roles)

        # if not regulated, consider as external input
        if n_parent_idx == 0:
            state_history[cur_step][idx] = state_prev[idx]
            continue

        # already processed
        if node_ready_time[idx] <= prev_step:
            state_history[cur_step][idx] = state_prev[idx]
            continue

        node_ready = False
        if np.max(node_ready_time[parents[:n_parent_idx]]) <= prev_step:
            node_ready = True
            node_ready_time[idx] = cur_step

        ## FIXME: reduce the impact of input values, ad hoc
        state_prev_shrink = np.copy(state_prev)
        # state_prev_shrink = (state_prev_shrink - hill_thr) / shrink_ratio
        ##

        if species == 0:  # transcription, TF -> mRNA
            f = calculate_promoter_activity(
                state_prev_shrink,
                parents,
                parents_roles,
                n_parent_idx,
                hill_thr,
                hill_coef,
            )
            # print(cur_step, idx, f)
            x = f * transcription_rate / mrna_degradation_rate
            if node_ready:
                noise = np.random.randn(1) * sigma
                noise_history[cur_step][idx] = noise
                xn = x + noise
                if xn < 0:
                    xn = 0
                state_history[cur_step][idx] = xn
            else:
                state_history[cur_step][idx] = x
        elif species == 1:  # translation, mRNA + miRNA -> protein
            mirna_activity, idx_mrna = calculate_translation_activity(
                state_prev_shrink,
                parents,
                parents_roles,
                n_parent_idx,
                hill_thr,
                hill_coef,
            )
            # miRNA reduce the translation rate
            p = (
                (translation_rate - mirna_activity)
                * state_prev_shrink[idx_mrna]
                / protein_degradation_rate
            )
            if node_ready:
                noise = np.random.randn(1) * sigma
                noise_history[cur_step][idx] = noise
                # noise = 0
                pn = p + noise
                if pn < 0:
                    pn = 0
                state_history[cur_step][idx] = pn
            else:
                state_history[cur_step][idx] = p
        else:
            raise ValueError


def run_sim(
    idx_layer,
    idx_parent,
    idx_parent_roles,
    sigma_in=0.5,
    sigma_mid=0.2,
    n_sample=1,
    n_max_steps=1000,
    method="steady",
    dt=0.02,
    hill_coef=1,
    shrink_ratio=4,
):
    # integrator for each molecule
    n_node = len(idx_layer)
    dat_network = np.zeros((n_sample, n_node))

    # make the parent and role dictionary arrays for faster calculation
    idx_layer = np.array(list(idx_layer.values()))
    n_parent = np.zeros(n_node, dtype=int)
    idx_parent_arr = np.zeros((n_node, 3), dtype=int) - 1
    for i, x in idx_parent.items():
        idx_parent_arr[i, : len(x)] = x
        n_parent[i] = len(x)
    idx_parent_roles_arr = np.zeros((n_node, 3), dtype=int) - 1
    for i, x in idx_parent_roles.items():
        idx_parent_roles_arr[i, : len(x)] = x

    state_history = []
    noise_history = []  # for debug
    for n in tqdm(range(n_sample)):
        noise_history = np.zeros((n_max_steps, n_node))

        # for steady state only
        # record the step when all parents are ready
        # if all parents are ready, add noise
        node_ready_time = np.zeros(n_node) + 10000000
        node_ready_time[n_parent == 0] = 0

        # Use a smaller range to be closer to the linear range
        # TODO: is this needed?
        xx = np.random.randn(n_node) * sigma_in + 10
        xx[xx < 0] = 0

        state_history = np.zeros((n_max_steps, n_node))
        state_history[0] = xx

        # state_history[0] = 10
        # state_history[0] = 10 + np.random.randn() * 0.5

        cur_step = 0
        for cur_step in range(1, n_max_steps):
            if method == "ode":
                ode_one_step(
                    state_history,
                    cur_step,
                    idx_layer,
                    idx_parent_arr,
                    idx_parent_roles_arr,
                    n_parent,
                    dt=dt,
                )
            elif method == "steady":
                steady_state_one_step(
                    state_history,
                    noise_history,
                    node_ready_time,
                    cur_step,
                    idx_layer,
                    idx_parent_arr,
                    idx_parent_roles_arr,
                    n_parent,
                    hill_coef=hill_coef,
                    sigma=sigma_mid,
                    # shrink_ratio=shrink_ratio,
                )
                if np.max(node_ready_time) < 10000000:
                    break
            else:
                raise ValueError

        if method == "ode":
            dat_network[n] = state_history[-1]
        elif method == "steady":
            dat_network[n] = state_history[cur_step]

    return dat_network, state_history, noise_history
