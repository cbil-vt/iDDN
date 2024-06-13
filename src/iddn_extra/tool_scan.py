import numpy as np
from tqdm import tqdm
import h5py
from ddn3 import ddn, tools, performance
from iddn import iddn


def scan_ddn(dat1, dat2, lambda1_rg, lambda2=0.1, mthd="resi"):
    t1_lst = []
    t2_lst = []
    for i, lamb in enumerate(lambda1_rg):
        out_ddn = ddn.ddn(
            dat1, dat2, lambda1=lamb, lambda2=lambda2, threshold=1e-5, mthd=mthd
        )
        t1_lst.append(np.copy(out_ddn[0]))
        t2_lst.append(np.copy(out_ddn[1]))
    return t1_lst, t2_lst


def scan_iddn(
    dat1,
    dat2,
    dep_mat,
    rho1_mat,
    rho2_mat,
    lambda1_rg,
    lambda2=0.1,
    mthd="resi",
):
    t1_lst = []
    t2_lst = []
    for i, lambda1 in enumerate(lambda1_rg):
        # print(i, lambda1)
        lambda1_mat = rho1_mat * lambda1
        lambda2_mat = rho2_mat * lambda2
        out_ddn = iddn.iddn(
            dat1,
            dat2,
            lambda1=lambda1_mat,
            lambda2=lambda2_mat,
            dep_mat=dep_mat,
            mthd=mthd,
        )
        t1_lst.append(np.copy(out_ddn[0]))
        t2_lst.append(np.copy(out_ddn[1]))
    return t1_lst, t2_lst


def scan_error_measure_per_condition(
    t1_lst, t2_lst, g1_gt, g2_gt, msk_out=None, msk_in=None
):
    res_g1 = np.zeros((len(t1_lst), 5))
    res_g2 = np.zeros((len(t1_lst), 5))

    if msk_out is None:
        msk_out = np.zeros_like(g1_gt)
    if msk_in is None:
        msk_in = np.ones_like(g1_gt)

    msk_in = msk_in + msk_in.T
    msk_in = 1 * (msk_in > 0)
    msk_out = msk_out + msk_out.T
    msk_out = 1 * (msk_out > 0)

    g1_gt[msk_out > 0] = 0
    g2_gt[msk_out > 0] = 0
    g1_gt = g1_gt * msk_in
    g2_gt = g2_gt * msk_in

    for i in range(len(t1_lst)):
        t1 = t1_lst[i]
        t2 = t2_lst[i]
        g1_est = tools.get_net_topo_from_mat(t1)
        g2_est = tools.get_net_topo_from_mat(t2)
        g1_est = g1_est * msk_in
        g2_est = g2_est * msk_in

        g1_est[msk_out > 0] = 0
        g2_est[msk_out > 0] = 0
        res_g1[i] = performance.get_error_measure_two_theta(g1_est, g1_gt)
        res_g2[i] = performance.get_error_measure_two_theta(g2_est, g2_gt)
    return res_g1, res_g2


def scan_error_measure_comm_diff(t1_lst, t2_lst, comm_gt, diff_gt, msk=None):
    # The mask may look like np.ix_(np.arange(2,5), np.arange(3,6))
    res_comm = np.zeros((len(t1_lst), 5))
    res_diff = np.zeros((len(t1_lst), 5))
    for i in range(len(t1_lst)):
        comm_est, diff_est = tools.get_common_diff_net_topo([t1_lst[i], t2_lst[i]])
        if msk is not None:
            res_comm[i] = performance.get_error_measure_two_theta(
                comm_est[msk], comm_gt[msk]
            )
            res_diff[i] = performance.get_error_measure_two_theta(
                diff_est[msk], diff_gt[msk]
            )
        else:
            res_comm[i] = performance.get_error_measure_two_theta(comm_est, comm_gt)
            res_diff[i] = performance.get_error_measure_two_theta(diff_est, diff_gt)
    return res_comm, res_diff


def scan2_ddn(dat1, dat2, rho1_rg, rho2_rg, n_sample_work, sigma_add=0, n=0):
    # Repeat, lambda1, lambda2, conditions, feature, feature
    n_sample, n_feature = dat1.shape
    idx1 = np.random.choice(n_sample, n_sample_work, replace=False)
    idx2 = np.random.choice(n_sample, n_sample_work, replace=False)
    dat1_sel = dat1[idx1, :]
    dat2_sel = dat2[idx2, :]
    dat1_sel = dat1_sel + np.random.normal(0, sigma_add, dat1_sel.shape)
    dat2_sel = dat2_sel + np.random.normal(0, sigma_add, dat2_sel.shape)
    res_mat0 = np.zeros((len(rho1_rg), len(rho2_rg), 2, n_feature, n_feature))

    for j, rho2 in enumerate(rho2_rg):
        print(n, rho2)
        t1_lst, t2_lst = scan_ddn(dat1_sel, dat2_sel, rho1_rg, lambda2=rho2)
        res_mat0[:, j, 0] = np.array(t1_lst)
        res_mat0[:, j, 1] = np.array(t2_lst)
    return res_mat0


def scan2_iddn(dat1, dat2, rho1_rg, rho2_rg, dep_mat=None, n_sample_work=100, n=0):
    # Repeat, lambda1, lambda2, conditions, feature, feature
    n_sample, n_feature = dat1.shape
    idx1 = np.random.choice(n_sample, n_sample_work, replace=False)
    idx2 = np.random.choice(n_sample, n_sample_work, replace=False)
    dat1_sel = dat1[idx1, :]
    dat2_sel = dat2[idx2, :]

    if dep_mat is None:
        dep_mat = np.ones((n_feature, n_feature))
    rho1_mat = np.ones((n_feature, n_feature))
    rho2_mat = np.ones((n_feature, n_feature))

    res_mat0 = np.zeros((len(rho1_rg), len(rho2_rg), 2, n_feature, n_feature))
    for j, rho2 in enumerate(rho2_rg):
        print(n, rho2)
        t1_lst, t2_lst = scan_iddn(
            dat1_sel,
            dat2_sel,
            dep_mat,
            rho1_mat,
            rho2_mat,
            rho1_rg,
            lambda2=rho2,
        )
        res_mat0[:, j, 0] = np.array(t1_lst)
        res_mat0[:, j, 1] = np.array(t2_lst)
    return res_mat0


def gather_res(h5_file, comm_gt, diff_gt, con_mat1, con_mat2, tt=True):
    # Repeat, lambda1, lambda2, conditions, feature, feature
    f = h5py.File(h5_file, "r")
    print(list(f.keys()))
    dep_est = np.array(f["dep_est"])
    f.close()
    if tt:
        dep_est = np.transpose(dep_est, np.arange(len(dep_est.shape) - 1, -1, -1))
    print(dep_est.shape)

    n_rep, n_rho1, n_rho2, _, _, _ = dep_est.shape

    res_comm_lst = np.zeros((n_rep, n_rho1, n_rho2, 5))
    res_g1_lst = np.zeros((n_rep, n_rho1, n_rho2, 5))

    for n in range(n_rep):
        print(n)
        for j in range(n_rho2):
            t1_lst = dep_est[n, :, j, 0]
            t2_lst = dep_est[n, :, j, 1]

            res_comm, res_diff = scan_error_measure_comm_diff(
                t1_lst, t2_lst, comm_gt, diff_gt
            )
            res_g1, res_g2 = scan_error_measure_per_condition(
                t1_lst, t2_lst, con_mat1, con_mat2
            )
            res_comm_lst[n, :, j] = res_comm
            res_g1_lst[n, :, j] = res_g1

    res_comm_mean = np.mean(res_comm_lst, axis=0)
    res_g1_mean = np.mean(res_g1_lst, axis=0)

    return res_comm_mean, res_g1_mean, dep_est
