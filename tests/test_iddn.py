# Test basic functions of iDDN without constraints

import numpy as np
from iddn import iddn
from iddn_data import load_data


temp = load_data.load_example("two_part_network.npz")
dat1 = temp["dat1"]
dat2 = temp["dat2"]
lambda1x = temp["lambda1"]
lambda2x = temp["lambda2"]
omega1_ref = temp["omega1"]
omega2_ref = temp["omega2"]

n_feature = dat1.shape[1]
dep_mat = np.ones((n_feature, n_feature))
lambda1 = dep_mat * lambda1x
lambda2 = dep_mat * lambda2x

ERR_THRESHOLD = 1e-8


def get_err(omega1, omega2, omega1_ref, omega2_ref):
    d1 = np.linalg.norm(omega1 - omega1_ref)
    d2 = np.linalg.norm(omega2 - omega2_ref)
    return d1, d2


def test_iddn_resi():
    omega1, omega2 = iddn.iddn(
        dat1,
        dat2,
        dep_mat=dep_mat,
        lambda1=lambda1,
        lambda2=lambda2,
        mthd="resi",
    )
    d1, d2 = get_err(omega1, omega2, omega1_ref, omega2_ref)
    assert d1 < ERR_THRESHOLD
    assert d2 < ERR_THRESHOLD


def test_iddn_corr():
    omega1, omega2 = iddn.iddn(
        dat1,
        dat2,
        dep_mat=dep_mat,
        lambda1=lambda1,
        lambda2=lambda2,
        mthd="corr",
    )
    d1, d2 = get_err(omega1, omega2, omega1_ref, omega2_ref)
    assert d1 < ERR_THRESHOLD
    assert d2 < ERR_THRESHOLD


def test_iddn_resi_parallel():
    omega1, omega2 = iddn.iddn_parallel(
        dat1,
        dat2,
        dep_mat=dep_mat,
        lambda1=lambda1,
        lambda2=lambda2,
        mthd="resi",
        n_process=2,
    )
    d1, d2 = get_err(omega1, omega2, omega1_ref, omega2_ref)
    assert d1 < ERR_THRESHOLD
    assert d2 < ERR_THRESHOLD


def test_iddn_corr_parallel():
    omega1, omega2 = iddn.iddn_parallel(
        dat1,
        dat2,
        dep_mat=dep_mat,
        lambda1=lambda1,
        lambda2=lambda2,
        mthd="corr",
        n_process=2,
    )
    d1, d2 = get_err(omega1, omega2, omega1_ref, omega2_ref)
    assert d1 < ERR_THRESHOLD
    assert d2 < ERR_THRESHOLD


if __name__ == "__main__":
    test_iddn_resi()
