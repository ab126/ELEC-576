import os

import tensorflow as tf
import numpy as np
import nibabel as nib

def load_nifti(nifti_file_path):
    """ Given NIFTI file path returns data matrix of shape (t_dim, x_dim, y_dim, z_dim) in MNI coordinates """
    f_img = nib.load(nifti_file_path)
    f_img_data = f_img.get_fdata()
    f_img_data = np.swapaxes(f_img_data,0, 1)
    f_img_data = np.moveaxis(f_img_data, -1, 0)
    return f_img_data


class NSDDataset(tf.keras.utils.PyDataset):
    """ Custom dataset handler. Traverse order is according to os package which appears to be random within subjects """

    def __init__(self, ppdata_path, batch_size = 10, **kwargs):
        """ Each batch is a section of a run in a session """
        super().__init__(**kwargs)
        self.ppdata_path = ppdata_path
        self.nifti_rel_path = '/func1pt8mm/timeseries'
        self.numpy_rel_path = '/func1pt8mm/numpy'

        idx = 0

        self.subject_inds = []
        self.n_batches = idx
        self.batch_size = batch_size # Number of samples in time
        self.subj_run_batch_sizes = {} # Number of batches in each subject run
        self.subj_run_batch_idxs = {}  # Starting indices of batches in each subject run

        # Dimensions of the fMRI
        self.max_x = 0
        self.max_y = 0
        self.max_z = 0

    def precompute_nifti2numpy(self, quick=True, only_indices=False):
        """ Converts nifti images into preprocessed (cropped, scaled) numpy arrays """

        # First pass to find dimensions
        run_idx = 0
        idx = 0 # batch_idx
        print('1st pass ')
        for subject in os.listdir(self.ppdata_path):
            # if subject != 'subj04':
            #     continue
            print('\nSubject {}'.format(subject))
            self.subj_run_batch_sizes[subject] = {}
            # self.subj_run_batch_idxs[subject] = {}

            numpy_path = self.ppdata_path + '/' + subject + self.numpy_rel_path
            if not os.path.exists(numpy_path):
                os.mkdir(numpy_path)

            nifti_path = self.ppdata_path + '/' + subject + self.nifti_rel_path
            for nifti_run in os.listdir(nifti_path):
                if run_idx % 10 == 0:
                    print(nifti_run)
                img_path = self.ppdata_path + '/' + subject + self.nifti_rel_path + '/' + nifti_run
                f_img_data = load_nifti(img_path)

                d_t = f_img_data.shape[0]
                if f_img_data.shape[1] > self.max_x:
                    self.max_x = f_img_data.shape[1]
                if f_img_data.shape[2] > self.max_y:
                    self.max_y = f_img_data.shape[2]
                if f_img_data.shape[3] > self.max_z:
                    self.max_z = f_img_data.shape[3]

                # Divide and save batch info
                run_n_batches = int(np.ceil(d_t / self.batch_size))
                self.subj_run_batch_sizes[subject][nifti_run] = run_n_batches
                self.subject_inds.append((subject, idx))

                idx += run_n_batches
                run_idx += 1
                if quick:
                    for nifti_run2 in os.listdir(nifti_path):
                        self.subj_run_batch_sizes[subject][nifti_run2] = run_n_batches
                    break
            print('Done')
        self.n_batches = idx

        if only_indices:
            return

        # Second pass
        print('\n2nd pass ')
        for subject in os.listdir(self.ppdata_path):
            # if subject != 'subj04': # TODO: Unlimit this after test
            #     continue
            print('\nSubject {}'.format(subject))
            numpy_path = self.ppdata_path + '/' + subject + self.numpy_rel_path
            nifti_path = self.ppdata_path + '/' + subject + self.nifti_rel_path

            run_idx = 0
            for nifti_run in os.listdir(nifti_path):
                # if run_idx > 10: # TODO: Unlimit this after test
                #     break
                img_path = self.ppdata_path + '/' + subject + self.nifti_rel_path + '/' + nifti_run
                f_img_data = load_nifti(img_path)

                # Process the nifti numpy array

                ## Edge pad to maximal dimension
                # Left to right
                width_x = self.max_x - f_img_data.shape[1]
                before_x = width_x // 2
                after_x = width_x - before_x

                # Posterior to anterior
                width_y = self.max_y - f_img_data.shape[2]
                before_y = width_y // 2
                after_y = width_y - before_y

                # Inferior to superior
                width_z = self.max_z - f_img_data.shape[3]
                before_z = width_z // 2
                after_z = width_z - before_z

                mat_x = np.pad(f_img_data, ((0, 0), (before_x, after_x), (before_y, after_y),
                                            (before_z, after_z)))

                # Make dimensions even & downcast for memory # TODO : clip symmetrically?
                d_t, d_x, d_y, d_z = mat_x.shape
                d_x = d_x//4 * 4
                d_y = d_y // 4 * 4
                d_z = d_z // 4 * 4
                mat_x = mat_x[:, :d_x, :d_y, :d_z]
                mat_x = mat_x.astype(np.float32)

                # Divide and save numpy array
                run_n_batches = self.subj_run_batch_sizes[subject][nifti_run]

                for j in range(run_n_batches):
                    arr_path = numpy_path + '/' + nifti_run + '_batch{}'.format(j) + '.npy'
                    batch_mat_x = mat_x[j * self.batch_size : (j + 1) * self.batch_size, ...]
                    np.save(arr_path, batch_mat_x)

                run_idx += 1
            print('Done')

    def __len__(self):
        """ Return number of batches """
        #return 200
        return self.n_batches

    def __getitem__(self, idx):
        """ Return mat_x and dummy variable y from batch """

        subject, rel_idx = self.find_subject_from_idx(idx)
        numpy_path = self.ppdata_path + '/' + subject + self.numpy_rel_path
        batch_name = os.listdir(numpy_path)[rel_idx] # TODO: Might be permuting batches here

        arr_path = numpy_path + '/' + batch_name
        mat_x = np.expand_dims(np.load(arr_path), axis=-1)
        return mat_x, mat_x

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

