import glob
import os

import tensorflow as tf
import numpy as np
import nibabel as nib

def sigmoid(x):
    return 1/(1.0 + np.exp(-x))

def load_nifti(nifti_file_path):
    """ Given NIFTI file path returns data matrix of shape (t_dim, x_dim, y_dim, z_dim) in MNI coordinates """
    f_img = nib.load(nifti_file_path)
    f_img_data = f_img.get_fdata()
    f_img_data = np.swapaxes(f_img_data,0, 1)
    f_img_data = np.moveaxis(f_img_data, -1, 0)

    # Normalize
    f_img_data -= np.mean(f_img_data)
    f_img_data /= np.std(f_img_data)
    f_img_data = sigmoid(f_img_data)

    return f_img_data


class fMRIDataset(tf.keras.utils.PyDataset):
    """ Custom dataset handler. Traverse order is according to glob package which appears to be random within subjects """

    def __init__(self, ppdata_path, nifti_rel_path, numpy_rel_path=None, batch_size = 10, batch_offset=0,
                 single_subject=None, private_dataset=False, **kwargs):
        """ Each batch is a section of a run in a session """
        super().__init__(**kwargs)
        self.ppdata_path = ppdata_path
        self.nifti_rel_path = nifti_rel_path
        self.numpy_rel_path = '/numpy' if numpy_rel_path is None else numpy_rel_path
        self.batch_offset = batch_offset
        self.single_subject = single_subject
        self.private_dataset = private_dataset

        self.subject_inds = []
        self.n_batches = 0
        self.batch_size = batch_size # Number of samples in time
        self.subj_run_batch_sizes = {} # Number of batches in each subject run
        self.subj_run_batch_idxs = {}  # Starting indices of batches in each subject run
        self.subj_tot_samples = {}

        # Dimensions of the fMRI
        self.max_x = 0
        self.max_y = 0
        self.max_z = 0

    def precompute_nifti2numpy(self, quick=True, only_indices=False, set_max_x=None,
                               set_max_y=None, set_max_z=None):
        """
        Converts nifti images into preprocessed (cropped, scaled) numpy arrays. If set_max params are given crops
        images accordingly
        """

        def first_pass():
            """ Compute the indices and max dimensions of nifti images"""
            run_idx = 0
            idx = 0  # batch_idx
            for subject_path in glob.glob(self.ppdata_path + '/*'): #os.listdir(self.ppdata_path):
                if not os.path.isdir(subject_path):
                    continue
                subject = subject_path.split('/')[-1]

                if self.single_subject is not None and subject != self.single_subject:
                    continue
                print('\nSubject {}'.format(subject))
                self.subj_run_batch_sizes[subject] = {}
                self.subj_tot_samples[subject] = 0
                # self.subj_run_batch_idxs[subject] = {}

                numpy_path = self.ppdata_path + '/' + subject + self.numpy_rel_path
                if not os.path.exists(numpy_path):
                    os.mkdir(numpy_path)
                nifti_path = self.ppdata_path + '/' + subject + self.nifti_rel_path

                # First pass: Indices and max dimensions
                for nifti_run_path in glob.glob(nifti_path + '/*.nii'):
                    nifti_run = nifti_run_path.split('/')[-1]

                    if run_idx % 10 == 0:
                        print(nifti_run)
                    img_path = self.ppdata_path + '/' + subject + self.nifti_rel_path + '/' + nifti_run
                    f_img_data = load_nifti(img_path)

                    d_t = f_img_data.shape[0]
                    if set_max_x is not None:
                        self.max_x = set_max_x
                    elif f_img_data.shape[1] > self.max_x:
                        self.max_x = f_img_data.shape[1]
                    if set_max_y is not None:
                        self.max_y = set_max_y
                    elif f_img_data.shape[2] > self.max_y:
                        self.max_y = f_img_data.shape[2]
                    if set_max_z is not None:
                        self.max_z = set_max_z
                    elif f_img_data.shape[3] > self.max_z:
                        self.max_z = f_img_data.shape[3]

                    # Divide and save batch info
                    run_n_batches = int(np.ceil(d_t / self.batch_size))
                    self.subj_run_batch_sizes[subject][nifti_run] = run_n_batches
                    self.subj_tot_samples[subject] += d_t
                    self.subject_inds.append((subject, idx))

                    idx += run_n_batches
                    run_idx += 1
                    if quick:
                        self.subj_tot_samples[subject] *= len(glob.glob(nifti_path + '/*.nii'))
                        for nifti_run_path2 in glob.glob(nifti_path + '/*.nii'):
                            nifti_run2 = nifti_run_path2.split('/')[-1]
                            self.subj_run_batch_sizes[subject][nifti_run2] = run_n_batches
                        idx += run_n_batches * (len(glob.glob(nifti_path + '/*.nii')) - 1)
                        break
                print('Done')
            self.n_batches = idx

        def second_pass():
            """ Computes the numpy arrays and saves into memory """

            for subject_path in glob.glob(self.ppdata_path + '/*'):  # os.listdir(self.ppdata_path):
                if not os.path.isdir(subject_path):
                    continue
                subject = subject_path.split('/')[-1]

                if self.single_subject is not None and subject != self.single_subject:
                    continue
                print('\nSubject {}'.format(subject))
                numpy_path = self.ppdata_path + '/' + subject + self.numpy_rel_path
                nifti_path = self.ppdata_path + '/' + subject + self.nifti_rel_path

                run_idx = 0
                for nifti_run_path in glob.glob(nifti_path + '/*.nii'):
                    nifti_run = nifti_run_path.split('/')[-1]
                    # if run_idx > 10: #
                    #     break
                    img_path = self.ppdata_path + '/' + subject + self.nifti_rel_path + '/' + nifti_run
                    f_img_data = load_nifti(img_path)

                    # Process the nifti numpy array

                    ## Edge pad to maximal dimension
                    # Left to right
                    pad_x = self.max_x - f_img_data.shape[1]
                    before_x = pad_x // 2
                    after_x = pad_x - before_x
                    if before_x < 0: # Happens iff pad_x < 0
                        f_img_data = f_img_data[:, -before_x:-before_x + self.max_x, :, :]
                        before_x, after_x = 0, 0

                    # Posterior to anterior
                    pad_y = self.max_y - f_img_data.shape[2]
                    before_y = pad_y // 2
                    after_y = pad_y - before_y
                    if before_y < 0: # Happens iff pad_y < 0
                        f_img_data = f_img_data[:, :, -before_y:-before_y + self.max_y, :]
                        before_y, after_y = 0, 0

                    # Inferior to superior
                    pad_z = self.max_z - f_img_data.shape[3]
                    before_z = pad_z // 2
                    after_z = pad_z - before_z
                    if before_z < 0: # Happens iff pad_z < 0
                        f_img_data = f_img_data[:, :, :, -before_z:-before_z + self.max_z]
                        before_z, after_z = 0, 0

                    mat_x = np.pad(f_img_data, ((0, 0), (before_x, after_x), (before_y, after_y),
                                                (before_z, after_z)))

                    # Downsample dimensions & down typecast for memory
                    d_t_orig, d_x_orig, d_y_orig, d_z_orig = mat_x.shape

                    d_x = d_x_orig // 8 * 8
                    off_x = (d_x_orig - d_x) // 2

                    d_y = d_y_orig // 8 * 8
                    off_y = (d_y_orig - d_y) // 2

                    d_z = d_z_orig // 8 * 8
                    off_z = (d_z_orig - d_z) // 2

                    mat_x = mat_x[:, off_x: off_x + d_x, off_y:off_y + d_y, off_z:off_z + d_z]
                    mat_x = mat_x.astype(np.float32)

                    # Divide and save numpy array
                    run_n_batches = self.subj_run_batch_sizes[subject][nifti_run]

                    for j in range(run_n_batches):
                        arr_path = numpy_path + '/' + nifti_run + '_batch{}'.format(j) + '.npy'
                        batch_mat_x = mat_x[j * self.batch_size: (j + 1) * self.batch_size, ...]
                        np.save(arr_path, batch_mat_x)

                    run_idx += 1
                print('Done')

        # First pass to find dimensions and starting index of each subject
        print('1st pass ')
        first_pass()

        if only_indices:
            return

        # Second pass: Convert images to arrays
        print('\n2nd pass ')
        second_pass()

    def __len__(self):
        """ Return number of batches """
        n_batch = 0
        for subject_path in glob.glob(self.ppdata_path + '/*'):  # os.listdir(self.ppdata_path):
            if not os.path.isdir(subject_path):
                continue
            subject = subject_path.split('/')[-1]

            if self.single_subject is not None and subject != self.single_subject:
                continue
            numpy_path = self.ppdata_path + '/' + subject + self.numpy_rel_path
            n_batch += len(os.listdir(numpy_path))
        return n_batch - self.batch_offset
        #return self.n_batches

    def set_len(self, n_batch):
        self.__len__ = lambda : n_batch

    def __getitem__(self, idx):
        """ Return mat_x and dummy variable y from batch """
        idx += self.batch_offset
        subject, rel_idx = self.find_subject_from_idx(idx)
        numpy_path = self.ppdata_path + '/' + subject + self.numpy_rel_path
        # print(rel_idx)
        # rel_idx = max(rel_idx, self.__len__() - 1)
        batch_name = os.listdir(numpy_path)[rel_idx] # TODO: Might be permuting batches here

        arr_path = numpy_path + '/' + batch_name
        mat_x = np.expand_dims(np.load(arr_path), axis=-1)

        if self.private_dataset:
            i = int(batch_name.split('_batch')[-1].split('.')[0])
            # print(batch_name)
            # tot_num = list(self.subj_run_batch_sizes[subject].values())[0]
            delta = 1 / self.subj_tot_samples[subject]
            y = i * self.batch_size * delta + np.arange(mat_x.shape[0]) * delta
            return mat_x, y

        return mat_x, mat_x

    def find_subject_from_idx(self, idx):
        """ Given idx finds the subject that has the respective batch """

        prev_idx = 0
        prev_subject = ''
        for subject, sub_idx in self.subject_inds:
            if idx < sub_idx:
                return prev_subject, idx - prev_idx
            prev_idx = sub_idx
            prev_subject = subject
        return self.subject_inds[-1][0], idx - self.subject_inds[-1][1]

