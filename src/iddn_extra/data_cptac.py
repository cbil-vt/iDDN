import numpy as np
import scipy.io as sio
import pandas as pd


class KinaseData:
    def __init__(
        self,
        dat_folder="../../../x_data/iddn/",
        cptac_file="CPTAC_Pros_MultiOmics_PNNL_Osama.mat",
        hrd_pos_file="HRD_20190421Update.txt",
        ksr_file="kinase_substrate.csv",
    ) -> None:
        # The full CPTAC data
        cptac_data = sio.loadmat(dat_folder + cptac_file)
        prot_dat_all, prot_samples, prot_names_all = self.get_full_protein_data(
            cptac_data
        )

        # The subset of kinase and substrates of interest in CPTAC data
        # A protein can be both kinase and substrate
        ksr_all = pd.read_csv(dat_folder + ksr_file).to_numpy()
        prot_dat, prot_name, node_kinase, node_substrate = self.get_ksr(
            ksr_all, prot_dat_all, prot_names_all
        )

        # The kinase-substrate interaction matrix for the subset of CPTAC data
        ksr_mat, ksr = self.get_ksr_mat(prot_name, ksr_all)

        # Split the data according to sample HRD
        f0 = dat_folder + hrd_pos_file
        hrd_pos_sample = pd.read_csv(f0, header=None)[0].to_list()
        dat_pos, dat_neg = self.split_data(prot_dat, hrd_pos_sample, prot_samples)

        self.ksr_all = ksr_all
        self.prot_names_all = [x[0][0] for x in prot_names_all]
    
        self.ksr = ksr
        self.ksr_mat = ksr_mat
        self.prot_dat_pos = dat_pos
        self.prot_dat_neg = dat_neg
        self.prot_name = prot_name
        self.node_kinase = node_kinase
        self.node_substrate = node_substrate

    def get_ksr(self, ksr_all, prot_dat_all, prot_names_all):
        node_kinase = []
        node_substrate = []
        for x in ksr_all:
            if not x[0] in node_kinase:
                node_kinase.append(x[0])
            if not x[1] in node_substrate:
                node_substrate.append(x[1])

        msk_prot = np.zeros(len(prot_names_all))
        prot_name = []
        for i in range(len(prot_names_all)):
            x = prot_names_all[i][0][0]
            if x in node_kinase or x in node_substrate:
                msk_prot[i] = 1
                prot_name.append(x)

        prot_dat = prot_dat_all[msk_prot > 0].T

        return prot_dat, prot_name, node_kinase, node_substrate

    def get_ksr_mat(self, prot_name, ksr_all):
        ksr = []
        for x in ksr_all:
            if x[0] in prot_name and x[1] in prot_name:
                ksr.append([x[0], x[1]])

        node_sel_idx_dict = dict()
        i = 0
        for x in prot_name:
            node_sel_idx_dict[x] = i
            i += 1

        n_node = len(prot_name)
        ksr_mat = np.zeros((n_node, n_node))
        for x in ksr:
            i = node_sel_idx_dict[x[0]]
            j = node_sel_idx_dict[x[1]]
            ksr_mat[i, j] = 1
            ksr_mat[j, i] = 1

        return ksr_mat, ksr

    def get_full_protein_data(self, cptac_data):
        prot = cptac_data["dataTumor"][0][0]
        prot_samples = [x[0] for x in prot[2][0]]
        prot_dat_all = prot[0]
        prot_names_all = prot[3]
        return prot_dat_all, prot_samples, prot_names_all

    def split_data(self, prot_dat, hrd_pos_sample, prot_samples):
        sample_hrd_mask = np.zeros(len(prot_samples))
        for i, x in enumerate(prot_samples):
            if x in hrd_pos_sample:
                sample_hrd_mask[i] = 1
        dat_pos = prot_dat[sample_hrd_mask > 0]
        dat_neg = prot_dat[sample_hrd_mask == 0]
        return dat_pos, dat_neg

    def print_protein_type(self):
        for x in self.prot_name:
            if x in self.node_kinase:
                print(x, "kinase")
            if x in self.node_substrate:
                print(x, "substrate")

    def print_kinase_substrate_overlap(self):
        print(set(self.node_kinase).intersection(set(self.node_substrate)))
