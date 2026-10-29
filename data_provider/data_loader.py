import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
# class Dataset_zero_shot(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  data_path='ETTh1.csv', target='OT'):  # 添加 seed 参数
#
#         assert flag in ['train', 'val', 'test']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#
#         self.root_path = root_path
#         self.data_path = data_path
#         self.target = target
#         self.__read_data__()
#
#     def __read_data__(self):
#         df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
#         cols = list(df_raw.columns)
#         cols.remove(self.target)
#
#         if 'date' in cols:
#             cols.remove('date')
#
#         self.data_x_full = df_raw[cols].values
#         self.data_y_full = df_raw[[self.target]].values
#         total_len = len(df_raw)
#
#         if self.set_type == 2:
#             self.data_x = self.data_x_full
#             self.data_y = self.data_y_full
#
#             self.sample_indices = np.arange(0, total_len - self.seq_len)
#
#         else:
#             self.data_x = self.data_x_full
#             self.data_y = self.data_y_full
#
#             all_train_val_indices = np.arange(0, len(self.data_x) - self.seq_len)
#
#             np.random.shuffle(all_train_val_indices)
#             num_samples = len(all_train_val_indices)
#             num_train_samples = int(num_samples * 0.8)
#
#             if self.set_type == 0:  # 'train'
#                 self.sample_indices = all_train_val_indices[:num_train_samples]
#             elif self.set_type == 1:  # 'val'
#                 self.sample_indices = all_train_val_indices[num_train_samples:]
#
#     def __getitem__(self, index):
#         s_begin = self.sample_indices[index]
#         s_end = s_begin + self.seq_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[s_end]
#         seq_x_mark = self.data_x[s_begin:s_end]
#         seq_y_mark = self.data_y[s_end]
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def __len__(self):
#         return len(self.sample_indices)


class Dataset_zero_shot(Dataset):
    """多工况软测量数据集：已知工况 7:2:1 时间切分，未知工况全部用于测试。

    目录约定（``root_path`` 下）::

        state1.csv, state2.csv, state3.csv       # 已知工况
        un_state1.csv, un_state2.csv             # 未知工况

    构建逻辑
    --------
    - 每个 CSV 单独按时间滑窗（stride=1），窗口不跨文件、**不打乱**；
    - 已知工况每个文件按窗口的时间顺序 7:2:1 切成 train / val / test（test 即
      "已知工况测试"），三段互不重叠；
    - 未知工况的全部窗口组成 ``flag='unknown'`` 数据集，用于零样本测试；
    - 目标列由 ``target`` 指定：本数据集的质量变量是 ``'11'``（物理上位于最后一列），
      其余 26 列作为工艺变量输入。

    单样本与 ``Dataset_zero_shot`` 一致，返回 ``(seq_x, seq_y, seq_x_mark, seq_y_mark)``，
    其中后两项是占位（与 ``seq_x`` / ``seq_y`` 相同），仅为兼容现有 exp 流程。
    """

    KNOWN_FILES = ('state1.csv', 'state2.csv', 'state3.csv')
    UNKNOWN_FILES = ('un_state1.csv', 'un_state2.csv')
    MULTI_FILE = True  # 由本类自己管理多个 CSV，data_factory 不再按 data_path 循环
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2  # train+val=0.9，剩余 0.1 作为已知工况测试集

    def __init__(self, root_path, flag='train', size=None, data_path=None,
                 target='11'):
        assert flag in ['train', 'val', 'test', 'unknown'], \
            f"flag 必须是 train/val/test/unknown，收到 {flag!r}"
        if size is None:
            raise ValueError("Dataset_hlt 需要 size=[seq_len, label_len, pred_len]")

        self.flag = flag
        self.seq_len, self.label_len, self.pred_len = size
        self.root_path = root_path
        self.target = str(target)

        self.data_x = []       # 每个文件的输入序列 [N, C]
        self.data_y = []       # 每个文件的目标序列 [N, 1]
        self.sample_files = []
        self.sample_starts = []
        self._read_data()

    # ------------------------------------------------------------------ 构建
    def _read_data(self):
        filenames = (
            self.UNKNOWN_FILES if self.flag == 'unknown' else self.KNOWN_FILES
        )

        for filename in filenames:
            path = os.path.join(self.root_path, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"找不到数据文件：{path}")

            df = pd.read_csv(path)
            if self.target not in df.columns:
                raise ValueError(
                    f"{filename} 中找不到目标列 {self.target!r}；"
                    f"现有列：{list(df.columns)}。"
                    "hlt 数据集的质量变量列名是 '11'（位于最后一列），请用 --target 11。"
                )

            # 目标列之外的列作为输入；没有时间列，date 若存在则丢弃
            x_cols = [c for c in df.columns if c != self.target and c != 'date']
            x = df[x_cols].values.astype(np.float32)
            y = df[[self.target]].values.astype(np.float32)

            n_windows = len(df) - self.seq_len
            if n_windows < 1:
                raise ValueError(
                    f"{filename} 只有 {len(df)} 行，小于等于 seq_len={self.seq_len}"
                )

            file_idx = len(self.data_x)
            self.data_x.append(x)
            self.data_y.append(y)

            starts = self._window_starts(n_windows)
            self.sample_files.extend([file_idx] * len(starts))
            self.sample_starts.extend(starts.tolist())

        self.sample_files = np.asarray(self.sample_files, dtype=np.int64)
        self.sample_starts = np.asarray(self.sample_starts, dtype=np.int64)

    def _window_starts(self, n_windows):
        """按时间顺序给出窗口起点，不做任何打乱。"""
        if self.flag == 'unknown':
            return np.arange(0, n_windows)

        train_end = int(n_windows * self.TRAIN_RATIO)
        val_end = int(n_windows * (self.TRAIN_RATIO + self.VAL_RATIO))
        if self.flag == 'train':
            return np.arange(0, train_end)
        if self.flag == 'val':
            return np.arange(train_end, val_end)
        return np.arange(val_end, n_windows)

    # ------------------------------------------------------------------ 取样
    def __getitem__(self, index):
        file_idx = int(self.sample_files[index])
        s_begin = int(self.sample_starts[index])
        s_end = s_begin + self.seq_len

        seq_x = self.data_x[file_idx][s_begin:s_end]
        seq_y = self.data_y[file_idx][s_end]
        # 时间特征占位，与输入/目标相同（现有模型不使用）
        return seq_x, seq_y, seq_x, seq_y

    def __len__(self):
        return len(self.sample_starts)
