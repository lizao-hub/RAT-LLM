import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# class Dataset_zero_shot(Dataset):
#     def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', target='OT'):
#
#         assert flag in ['train', 'val', 'test']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         self.root_path = root_path
#         self.data_path = data_path
#         self.target = target
#         self.__read_data__()
#
#     def __read_data__(self):
#         df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
#         cols = list(df_raw.columns)
#         # print(cols)
#         cols.remove(self.target)
#
#         if 'date' in cols:
#             cols.remove('date')
#         elif 'timestamp' in cols:
#             cols.remove('timestamp')
#
#         num_train = int(len(df_raw) * 0.7)
#         border1s = [0, num_train - self.seq_len, int(len(df_raw) * 0.8)-self.seq_len]
#         border2s = [num_train, len(df_raw), len(df_raw)]
#
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         data = df_raw[cols].values
#         target = df_raw[[self.target]].values
#         self.data_x = data[border1:border2]
#         self.data_y = target[border1:border2]
#
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[s_end]
#         seq_x_mark = self.data_x[s_begin:s_end]
#         seq_y_mark = self.data_y[s_end]
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len
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
        # self.seed = seed  # 保存 seed
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        cols = list(df_raw.columns)
        cols.remove(self.target)

        if 'date' in cols:
            cols.remove('date')
        elif 'timestamp' in cols:
            cols.remove('timestamp')

        # 加载所有数据，先不切分
        self.data_x_full = df_raw[cols].values
        self.data_y_full = df_raw[[self.target]].values

        total_len = len(df_raw)

        # --- 测试集逻辑 (set_type == 2) ---
        if self.set_type == 2:
            # # 测试集的数据块（包含 seq_len 重叠）
            # border1 = int(total_len * 0.8) - self.seq_len
            # border2 = total_len

            # self.data_x 和 self.data_y 是 __getitem__ 将要切片的连续数据块
            self.data_x = self.data_x_full
            self.data_y = self.data_y_full

            # 索引范围是从 0 到 (数据块长度 - seq_len)
            self.sample_indices = np.arange(0, total_len - self.seq_len)

        # --- 训练/验证集逻辑 (set_type == 0 or 1) ---
        else:

            # self.data_x 和 self.data_y 是 __getitem__ 将要切片的连续数据块
            self.data_x = self.data_x_full
            self.data_y = self.data_y_full

            # 生成这个数据块中 *所有* 可能的起始索引
            all_train_val_indices = np.arange(0, len(self.data_x) - self.seq_len)

            # --- 核心修改：打乱索引 ---
            # print(f"Shuffling train/val indices with seed {self.seed}")
            # np.random.seed(self.seed)
            np.random.shuffle(all_train_val_indices)

            # 按照 7:3 的比例 (70% vs 30%) 分割 *打乱后的索引列表*
            num_samples = len(all_train_val_indices)
            num_train_samples = int(num_samples * 0.8)

            if self.set_type == 0:  # 'train'
                self.sample_indices = all_train_val_indices[:num_train_samples]
            elif self.set_type == 1:  # 'val'
                self.sample_indices = all_train_val_indices[num_train_samples:]

    def __getitem__(self, index):
        # index 是 self.sample_indices 列表中的位置 (0, 1, 2...)
        # s_begin 是数据块中的 *实际* 起始位置
        s_begin = self.sample_indices[index]

        s_end = s_begin + self.seq_len

        # 从 self.data_x 数据块中切片
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end]  # 保持你原来的逻辑
        seq_x_mark = self.data_x[s_begin:s_end]  # 保持你原来的逻辑
        seq_y_mark = self.data_y[s_end]  # 保持你原来的逻辑

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # 返回当前数据集持有的 *样本* 数量
        return len(self.sample_indices)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', target='OT'):

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
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        if 'date' in cols:
            cols.remove('date')
        elif 'timestamp' in cols:
            cols.remove('timestamp')

        num_train = int(len(df_raw) * 0.6)
        num_val = int(len(df_raw) * 0.2)

        border1s = [0, num_train - self.seq_len, num_train + num_val - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        data = df_raw[cols].values
        target = df_raw[[self.target]].values
        self.data_x = data[border1:border2]
        self.data_y = target[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end ]
        seq_x_mark = self.data_x[s_begin:s_end]
        seq_y_mark = self.data_y[s_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len

