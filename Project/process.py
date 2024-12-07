import os

import tensorflow as tf
import numpy as np
import nibabel as nib

def load_nifti(nifti_file_path):
    """ Given NIFTI file path returns data matrix of shape (x_dim, y_dim, z_dim, t_dim) """
    f_img = nib.load(nifti_file_path)
    f_img_data = f_img.get_fdata()
    return f_img_data


class NSDDataset(tf.keras.utils.PyDataset):
    """ Custom dataset handler. Traverse order is according to os package which appears to be random within subjects """

    def __init__(self, ppdata_path, **kwargs):
        """ Each batch is a run in a session """
        super().__init__(**kwargs)
        self.ppdata_path = ppdata_path
        self.nifti_rel_path = '/func1pt8mm/timeseries'

        idx = 0
        subject_inds = []
        # Traverse all paths once and mark indices per subject
        for subject in os.listdir(self.ppdata_path):
            nifti_path = self.ppdata_path + '/' + subject + self.nifti_rel_path
            n_subj_batches = len(os.listdir(nifti_path))
            subject_inds.append((subject, idx))
            idx += n_subj_batches

        self.subject_inds = subject_inds
        self.n_batches = idx

    def __len__(self):
        """ Return number of batches """
        return self.n_batches

    def __getitem__(self, idx):
        """ Return mat_x and dummy variable y from batch """

        subject, rel_idx = self.find_subject_from_idx(idx)
        nifti_path = self.ppdata_path + '/' + subject + self.nifti_rel_path
        img_file = os.listdir(nifti_path)[rel_idx]
        img_path = nifti_path + '/' + img_file

        mat_x = load_nifti(img_path)
        return mat_x, np.zeros(mat_x.shape[-1])

    def find_subject_from_idx(self, idx):
        """ Given idx finds the subject that has the respective batch """

        prev_idx = 0
        prev_subject = 'subj01'
        for subject, sub_idx in self.subject_inds:
            if idx < sub_idx:
                return prev_subject, idx - prev_idx
            prev_idx = sub_idx
            prev_subject = subject
        return self.subject_inds[-1][0], idx - self.subject_inds[-1][1]
