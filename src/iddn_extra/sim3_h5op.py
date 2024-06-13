import numpy as np
import h5py
from ddn3 import tools


def make_new_sim_data(
    dat_file: str,
    dat1,
    dat2,
    con_mat1,
    con_mat2,
    comm_gt=None,
    diff_gt=None,
    dep_mat_null=None,
    dep_mat_prior=None,
    mol_layer=None,
):
    n_feature = dat1.shape[1]

    if dep_mat_null is None:
        dep_mat_null = np.ones((n_feature, n_feature))
    if dep_mat_prior is None:
        dep_mat_prior = np.ones((n_feature, n_feature))
    if comm_gt is None or diff_gt is None:
        comm_gt, diff_gt = tools.get_common_diff_net_topo([con_mat1, con_mat2])
    if mol_layer is None:
        mol_layer = [f"node_{i}" for i in range(n_feature)]

    f = h5py.File(dat_file, "w")
    f.create_dataset("dat1", data=dat1, compression="gzip")
    f.create_dataset("dat2", data=dat2, compression="gzip")
    f.create_dataset("dep_mat_null", data=dep_mat_null, compression="gzip")
    f.create_dataset("dep_mat_prior", data=dep_mat_prior, compression="gzip")
    f.create_dataset("con_mat1", data=con_mat1, compression="gzip")
    f.create_dataset("con_mat2", data=con_mat2, compression="gzip")
    f.create_dataset("comm_gt", data=comm_gt, compression="gzip")
    f.create_dataset("diff_gt", data=diff_gt, compression="gzip")
    f.create_dataset("node_names", data=mol_layer, compression="gzip")
    f.close()


def read_sim_data(dat_file, sigma=0):
    f = h5py.File(dat_file, "r")
    dat1 = np.array(f["dat1"])
    dat2 = np.array(f["dat2"])
    dep_mat_null = np.array(f["dep_mat_null"])
    dep_mat_prior = np.array(f["dep_mat_prior"])
    con_mat1 = np.array(f["con_mat1"])
    con_mat2 = np.array(f["con_mat2"])
    comm_gt = np.array(f["comm_gt"])
    diff_gt = np.array(f["diff_gt"])
    f.close()

    return dat1, dat2, con_mat1, con_mat2, comm_gt, diff_gt, dep_mat_null, dep_mat_prior
