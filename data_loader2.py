import numpy as np
import os
import pickle
import psutil
from keras.utils import Sequence

class CustomRealDataLoader(Sequence):
    def __init__(self, task, data_path, data_type, validation=False, batch_size=4, shuffle=True, multi_task=False):
        self.task = task
        if self.task == 'reach' or self.task == 'push':
            self.action_dim = 8
        elif self.task == 'pick':
            self.action_dim = 12
        elif self.task == 'place':
            self.action_dim = 10
        if multi_task:
            self.action_dim = 12

        self.data_path = data_path
        all_files_list = [f for f in os.listdir(data_path) if task in f and data_type in f]
        for f in all_files_list:
            if '%s_a_'%task in f:
                self.a_fname = f
            elif '%s_s_'%task in f:
                self.s_fname = f

        self.data_type = data_type
        if self.data_type == 'pkl':
            self.load_data = self.load_pkl
        elif self.data_type == 'npy':
            self.load_data = self.load_npy
        self.bs = batch_size

        self.a_list = self.load_data(self.a_fname)
        self.s_list = self.load_data(self.s_fname)
        assert len(self.a_list) == len(self.s_list)

        if validation:
            self.file_size = int(0.1 * len(self.a_list))
            self.a_list = self.a_list[-self.file_size:]
            self.s_list = self.s_list[-self.file_size:]
        else:
            self.file_size = int(0.9 * len(self.a_list))
            self.a_list = self.a_list[:self.file_size]
            self.s_list = self.s_list[:self.file_size]

        self.shuffle = shuffle
        self.validation = validation

    def __len__(self):
        return self.file_size // self.bs

    def __getitem__(self, idx):
        batch_action = self.a_list[idx * self.bs : (idx+1) * self.bs]
        batch_state = self.s_list[idx * self.bs : (idx+1) * self.bs]
        return batch_state, batch_action


    def load_pkl(self, pkl_file):
        with open(os.path.join(self.data_path, pkl_file), 'rb') as f:
            data = pickle.load(f)
        if self.task+'_s_' in pkl_file:
            d_0 = data[:, 0, :, :, :].copy()
            d_1 = data[:, 1, :, :, :].copy()
            data = np.concatenate([d_0, d_1], axis=-1)
        elif self.task + '_a_' in pkl_file:
            data = np.eye(self.action_dim)[data]
        return data

    def load_npy(self, npy_file):
        data = np.load(os.path.join(self.data_path, npy_file))
        if self.task + '_s_' in npy_file:
            d_0 = data[:, 0, :, :, :].copy()
            d_1 = data[:, 1, :, :, :].copy()
            data = np.concatenate([d_0, d_1], axis=-1)
        elif self.task + '_a_' in npy_file:
            data = np.eye(self.action_dim)[data]
        return data

    def load_valid_data(self):
        return self.s_list, self.a_list

    def on_epoch_end(self):
        if self.shuffle:
            shuffler = np.random.permutation(self.file_size)
            self.a_list = self.a_list[shuffler]
            self.s_list = self.s_list[shuffler]


class CustomDataLoader(Sequence):
    def __init__(self, task, data_path, data_type, validation=False, batch_size=256, file_size=512, shuffle=True, multi_task=False, t_idx=0, n_files=0, n_tasks=0):
        self.task = task
        if self.task == 'reach' or self.task == 'push':
            self.action_dim = 8
        elif self.task == 'pick':
            self.action_dim = 12
        elif self.task == 'place':
            self.action_dim = 10
        if multi_task:
            self.action_dim = 12 * n_tasks

        self.data_path = data_path
        all_files_list = [f for f in os.listdir(data_path) if task in f and data_type in f]
        self.a_list = sorted([f for f in all_files_list if '%s_a_'%task in f and data_type in f])
        self.s_list = sorted([f for f in all_files_list if '%s_s_' % task in f and data_type in f])
        # self.d_list = [f for f in all_files_list if '%s_d_' % task in f and data_type in f]
        assert len(self.a_list)==len(self.s_list)

        if n_files>0:
            n1 = n_files//2
            n2 = n_files - n_files//2
            self.a_list = self.a_list[:n1] + self.a_list[-n2-1:]
            self.s_list = self.s_list[:n1] + self.s_list[-n2-1:]
            print(self.a_list)
            # self.d_list = self.d_list[:n1] + self.d_list[-n2-1:]

        self.data_type = data_type
        if self.data_type == 'pkl':
            self.load_data = self.load_pkl
        elif self.data_type == 'npy':
            self.load_data = self.load_npy
        self.bs = batch_size
        self.file_size = file_size
        self.t_idx = t_idx
        self.shuffle = shuffle
        self.validation = validation

        self.pkl_idx = -1
        self.buff_a = np.array([])
        self.buff_s = np.array([])
        # self.buff_d = np.array([])

    def __len__(self):
        if self.validation:
            return self.file_size//self.bs
        else:
            return (self.file_size//self.bs) * (len(self.a_list) - 1)

    def __getitem__(self, idx):
        # process = psutil.Process(os.getpid())
        # print(idx, process.memory_percent())
        if self.validation:
            self.buff_a = self.load_data(self.a_list[-1])
            self.buff_s = self.load_data(self.s_list[-1])
            # self.buff_d = self.load_data(self.d_list[-1])
        else:
            pkl_idx = idx // (self.file_size//self.bs)
            if pkl_idx != self.pkl_idx:
                self.pkl_idx = pkl_idx
                self.buff_a = self.load_data(self.a_list[pkl_idx])
                self.buff_s = self.load_data(self.s_list[pkl_idx])
                # self.buff_d = self.load_data(self.d_list[pkl_idx])
                if False and self.shuffle:
                    shuffler = np.random.permutation(len(self.buff_a))
                    self.buff_a = self.buff_a[shuffler]
                    self.buff_s = self.buff_s[shuffler]
                    # self.buff_d = self.buff_d[shuffler]
        b_idx = idx % (self.file_size//self.bs)
        batch_action = self.buff_a[self.bs * b_idx: self.bs * (b_idx + 1)].copy()
        batch_state = self.buff_s[self.bs * b_idx: self.bs * (b_idx + 1)].copy()
        # batch_done = self.buff_d[self.bs * b_idx: self.bs * (b_idx + 1)]
        return batch_state, batch_action

    def load_pkl(self, pkl_file):
        with open(os.path.join(self.data_path, pkl_file), 'rb') as f:
            data = pickle.load(f)
        if self.task+'_s_' in pkl_file:
            d_0 = data[:, 0, :, :, :].copy()
            d_1 = data[:, 1, :, :, :].copy()
            data = np.concatenate([d_0, d_1], axis=-1)
        elif self.task + '_a_' in pkl_file:
            data = np.eye(self.action_dim)[data + self.t_idx * 12]
        return data

    def load_npy(self, npy_file):
        data = np.load(os.path.join(self.data_path, npy_file))
        if self.task + '_s_' in npy_file:
            d_0 = data[:, 0, :, :, :].copy()
            d_1 = data[:, 1, :, :, :].copy()
            data = np.concatenate([d_0, d_1], axis=-1)
        elif self.task + '_a_' in npy_file:
            data = np.eye(self.action_dim)[data + self.t_idx * 12]
        return data

    def load_valid_data(self):
        batch_a = self.load_data(self.a_list[-1])
        batch_s = self.load_data(self.s_list[-1])
        # batch_d = self.load_data(self.d_list[-1])
        return batch_s, batch_a

    def on_epoch_end(self):
        if self.shuffle:
            zipped_list = list(zip(self.a_list, self.s_list))
            # zipped_list = list(zip(self.a_list, self.s_list, self.d_list))
            np.random.shuffle(zipped_list)
            self.a_list, self.s_list = zip(*zipped_list)
            # self.a_list, self.s_list, self.d_list = zip(*zipped_list)


class CustomDataLoaderRotate(Sequence):
    def __init__(self, task, data_path, data_type, validation=False, batch_size=256, file_size=512, shuffle=True, multi_task=False, t_idx=0, n_files=0, n_tasks=0):
        self.task = task
        if self.task == 'reach' or self.task == 'push':
            self.action_dim = 8
        elif self.task == 'pick':
            self.action_dim = 12
        elif self.task == 'place':
            self.action_dim = 10
        if multi_task:
            self.action_dim = 12 * n_tasks

        self.data_path = data_path
        all_files_list = [f for f in os.listdir(data_path) if task in f and data_type in f]
        self.a_list = sorted([f for f in all_files_list if '%s_a_'%task in f and data_type in f])
        self.s_list = sorted([f for f in all_files_list if '%s_s_' % task in f and data_type in f])
        # self.d_list = [f for f in all_files_list if '%s_d_' % task in f and data_type in f]
        assert len(self.a_list)==len(self.s_list)

        if n_files>0:
            n1 = n_files//2
            n2 = n_files - n_files//2
            self.a_list = self.a_list[:n1] + self.a_list[-n2-1:]
            self.s_list = self.s_list[:n1] + self.s_list[-n2-1:]
            print(self.a_list)
            # self.d_list = self.d_list[:n1] + self.d_list[-n2-1:]

        self.data_type = data_type
        if self.data_type == 'pkl':
            self.load_data = self.load_pkl
        elif self.data_type == 'npy':
            self.load_data = self.load_npy
        self.bs = batch_size
        self.file_size = file_size
        self.t_idx = t_idx
        self.shuffle = shuffle
        self.validation = validation

        self.pkl_idx = -1
        self.buff_a = np.array([])
        self.buff_s = np.array([])
        # self.buff_d = np.array([])

    def __len__(self):
        if self.validation:
            return self.file_size//self.bs
        else:
            return (self.file_size//self.bs) * (len(self.a_list) - 1)

    def __getitem__(self, idx):
        # process = psutil.Process(os.getpid())
        # print(idx, process.memory_percent())
        if self.validation:
            self.buff_a = self.load_data(self.a_list[-1])
            self.buff_s = self.load_data(self.s_list[-1])
            # self.buff_d = self.load_data(self.d_list[-1])
        else:
            pkl_idx = idx // (self.file_size//self.bs)
            if pkl_idx != self.pkl_idx:
                self.pkl_idx = pkl_idx
                self.buff_a = self.load_data(self.a_list[pkl_idx])
                self.buff_s = self.load_data(self.s_list[pkl_idx])
                # self.buff_d = self.load_data(self.d_list[pkl_idx])
                if False and self.shuffle:
                    shuffler = np.random.permutation(len(self.buff_a))
                    self.buff_a = self.buff_a[shuffler]
                    self.buff_s = self.buff_s[shuffler]
                    # self.buff_d = self.buff_d[shuffler]
        b_idx = idx % (self.file_size//self.bs)
        batch_action = self.buff_a[self.bs * b_idx: self.bs * (b_idx + 1)].copy()
        batch_state = self.buff_s[self.bs * b_idx: self.bs * (b_idx + 1)].copy()
        # batch_done = self.buff_d[self.bs * b_idx: self.bs * (b_idx + 1)]
        return batch_state, batch_action

    def load_pkl(self, pkl_file):
        with open(os.path.join(self.data_path, pkl_file), 'rb') as f:
            data = pickle.load(f)
        if self.task+'_s_' in pkl_file:
            d_0 = data[:, 0, :, :, :].copy()
            d_1 = data[:, 1, :, :, :].copy()
            data = np.concatenate([d_0, d_1], axis=-1)
            for i in range(self.t_idx):
                data = np.rot90(data, axes=[1, 2])
        elif self.task + '_a_' in pkl_file:
            data = np.eye(self.action_dim)[data + self.t_idx * 12]
        return data

    def load_npy(self, npy_file):
        data = np.load(os.path.join(self.data_path, npy_file))
        if self.task + '_s_' in npy_file:
            d_0 = data[:, 0, :, :, :].copy()
            d_1 = data[:, 1, :, :, :].copy()
            data = np.concatenate([d_0, d_1], axis=-1)
            for i in range(self.t_idx):
                data = np.rot90(data, axes=[1, 2])
        elif self.task + '_a_' in npy_file:
            data = np.eye(self.action_dim)[data + self.t_idx * 12]
        return data

    def load_valid_data(self):
        batch_a = self.load_data(self.a_list[-1])
        batch_s = self.load_data(self.s_list[-1])
        # batch_d = self.load_data(self.d_list[-1])
        return batch_s, batch_a

    def on_epoch_end(self):
        if self.shuffle:
            zipped_list = list(zip(self.a_list, self.s_list))
            # zipped_list = list(zip(self.a_list, self.s_list, self.d_list))
            np.random.shuffle(zipped_list)
            self.a_list, self.s_list = zip(*zipped_list)
            # self.a_list, self.s_list, self.d_list = zip(*zipped_list)


class CustomDataLoaderV3(Sequence):
    def __init__(self, task, data_path, data_type, validation=False, batch_size=256, file_size=512, shuffle=True, multi_task=False, t_idx=0, n_files=0, n_tasks=0):
        self.task = task
        if self.task == 'reach' or self.task == 'push':
            self.action_dim = 8
        elif self.task == 'pick':
            self.action_dim = 12
        elif self.task == 'place':
            self.action_dim = 10
        if multi_task:
            self.action_dim = 12 * n_tasks

        self.data_path = data_path
        all_files_list = [f for f in os.listdir(data_path) if task in f and data_type in f]
        self.a_list = sorted([f for f in all_files_list if '%s_a_'%task in f and data_type in f])
        self.s_list = sorted([f for f in all_files_list if '%s_s_' % task in f and data_type in f])
        # self.d_list = [f for f in all_files_list if '%s_d_' % task in f and data_type in f]
        assert len(self.a_list)==len(self.s_list)

        if n_files>0:
            n1 = n_files//2
            n2 = n_files - n_files//2
            self.a_list = self.a_list[:n1] + self.a_list[-n2-1:]
            self.s_list = self.s_list[:n1] + self.s_list[-n2-1:]
            print(self.a_list)
            # self.d_list = self.d_list[:n1] + self.d_list[-n2-1:]

        self.data_type = data_type
        if self.data_type == 'pkl':
            self.load_data = self.load_pkl
        elif self.data_type == 'npy':
            self.load_data = self.load_npy
        self.bs = batch_size
        self.file_size = file_size
        self.t_idx = t_idx
        self.shuffle = shuffle
        self.validation = validation

        self.pkl_idx = -1
        self.buff_a = np.array([])
        self.buff_s = np.array([])
        # self.buff_d = np.array([])

    def __len__(self):
        if self.validation:
            return self.file_size//self.bs
        else:
            return (self.file_size//self.bs) * (len(self.a_list) - 1)

    def __getitem__(self, idx):
        # process = psutil.Process(os.getpid())
        # print(idx, process.memory_percent())
        if self.validation:
            self.buff_a = self.load_data(self.a_list[-1])
            self.buff_s = self.load_data(self.s_list[-1])
            # self.buff_d = self.load_data(self.d_list[-1])
        else:
            pkl_idx = idx // (self.file_size//self.bs)
            if pkl_idx != self.pkl_idx:
                self.pkl_idx = pkl_idx
                self.buff_a = self.load_data(self.a_list[pkl_idx])
                self.buff_s = self.load_data(self.s_list[pkl_idx])
                # self.buff_d = self.load_data(self.d_list[pkl_idx])
                if False and self.shuffle:
                    shuffler = np.random.permutation(len(self.buff_a))
                    self.buff_a = self.buff_a[shuffler]
                    self.buff_s = self.buff_s[shuffler]
                    # self.buff_d = self.buff_d[shuffler]
        b_idx = idx % (self.file_size//self.bs)
        batch_action = self.buff_a[self.bs * b_idx: self.bs * (b_idx + 1)].copy()
        batch_state = self.buff_s[self.bs * b_idx: self.bs * (b_idx + 1)].copy()
        # batch_done = self.buff_d[self.bs * b_idx: self.bs * (b_idx + 1)]
        return batch_state, batch_action

    def load_pkl(self, pkl_file):
        with open(os.path.join(self.data_path, pkl_file), 'rb') as f:
            data = pickle.load(f)
        if self.task+'_s_' in pkl_file:
            d_0 = data[:, 0, :, :, :].copy()
            d_1 = data[:, 1, :, :, :].copy()
            data = np.concatenate([d_0, d_1], axis=-1)
        elif self.task + '_a_' in pkl_file:
            data = np.eye(self.action_dim)[data + self.t_idx * 12]

        indexer = np.arange(-1,1)[None, :] + np.arange(self.bs)[:, None]
        indexer[0][0] = 0
        data = data[indexer]
        data = np.concatenate([data[:, 0], data[:, 1]], axis=-1)
        return data

    def load_npy(self, npy_file):
        data = np.load(os.path.join(self.data_path, npy_file))
        if self.task + '_s_' in npy_file:
            d_0 = data[:, 0, :, :, :].copy()
            d_1 = data[:, 1, :, :, :].copy()
            data = np.concatenate([d_0, d_1], axis=-1)
        elif self.task + '_a_' in npy_file:
            data = np.eye(self.action_dim)[data + self.t_idx * 12]
        indexer = np.arange(-1, 1)[None, :] + np.arange(self.bs)[:, None]
        indexer[0][0] = 0
        data = data[indexer]
        data = np.concatenate([data[:, 0], data[:, 1]], axis=-1)
        return data

    def load_valid_data(self):
        batch_a = self.load_data(self.a_list[-1])
        batch_s = self.load_data(self.s_list[-1])
        # batch_d = self.load_data(self.d_list[-1])

        batch_s = np.concatenate([batch_s[:, 0], batch_s[:, 1]], axis=-1)
        batch_a = np.concatenate([batch_a[:, 0], batch_a[:, 1]], axis=-1)
        return batch_s, batch_a

    def on_epoch_end(self):
        if self.shuffle:
            zipped_list = list(zip(self.a_list, self.s_list))
            # zipped_list = list(zip(self.a_list, self.s_list, self.d_list))
            np.random.shuffle(zipped_list)
            self.a_list, self.s_list = zip(*zipped_list)
            # self.a_list, self.s_list, self.d_list = zip(*zipped_list)
