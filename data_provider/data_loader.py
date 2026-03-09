import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class Dataset_zero_shot(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTh1.csv', target='OT'):  # 添加 seed 参数

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.root_path = root_path
        self.data_path = data_path
        self.target = target
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        cols = list(df_raw.columns)
        cols.remove(self.target)

        if 'date' in cols:
            cols.remove('date')

        self.data_x_full = df_raw[cols].values
        self.data_y_full = df_raw[[self.target]].values
        total_len = len(df_raw)

        if self.set_type == 2:
            self.data_x = self.data_x_full
            self.data_y = self.data_y_full

            self.sample_indices = np.arange(0, total_len - self.seq_len)

        else:
            self.data_x = self.data_x_full
            self.data_y = self.data_y_full

            all_train_val_indices = np.arange(0, len(self.data_x) - self.seq_len)

            np.random.shuffle(all_train_val_indices)
            num_samples = len(all_train_val_indices)
            num_train_samples = int(num_samples * 0.8)

            if self.set_type == 0:  # 'train'
                self.sample_indices = all_train_val_indices[:num_train_samples]
            elif self.set_type == 1:  # 'val'
                self.sample_indices = all_train_val_indices[num_train_samples:]

    def __getitem__(self, index):
        s_begin = self.sample_indices[index]
        s_end = s_begin + self.seq_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end]
        seq_x_mark = self.data_x[s_begin:s_end]
        seq_y_mark = self.data_y[s_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.sample_indices)