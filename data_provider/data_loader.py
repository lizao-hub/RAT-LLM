import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class Dataset_Custom(Dataset):
    """单文件软测量数据集：一个 CSV 按时间 7:1:2 切 train / val / test。

    样本定义
    --------
    - 输入 ``seq_x``：``[t - seq_len + 1, t]`` 共 ``seq_len`` 行的工艺变量；
    - 标签 ``seq_y``：第 ``t`` 行的目标变量，即窗口**最后一个时刻**的值（同一时刻估计）。

    划分方式
    --------
    先按行数把文件切成 70% / 10% / 20% 三段，再让每段的**标签行**落在自己的区间内::

        train 标签行  [seq_len - 1, 0.7N)
        val   标签行  [0.7N,       0.8N)
        test  标签行  [0.8N,       N)

    三段标签互不重叠。旧写法 ``border1 = num_train - seq_len`` 会让边界处的同一个
    标签同时属于 val 和 test（反过来也一样），这里改成按标签位置划分就消除了。
    输入窗口允许向前跨越段边界去借历史行，但标签绝不跨段，因此不会泄漏。

    样本数
    ------
    train = ``0.7N - seq_len + 1``（前 ``seq_len - 1`` 行没有完整历史，构不成样本），
    val = ``0.1N``，test = ``0.2N``，正好对得上行数比例。
    """

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1  # test 比例 = 1 - 0.7 - 0.1 = 0.2

    def __init__(self, root_path, data_path='ETTh1.csv', flag='train', size=None, target='OT'):
        # size = [seq_len]
        # if size is None:
        #     raise ValueError('Dataset_Custom 需要 size=[seq_len]')
        self.seq_len = int(size[0])

        assert flag in ['train', 'val', 'test'], f'flag 必须是 train/val/test，收到 {flag!r}'
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
        df_raw.columns: [...(process variables), target feature]
        '''
        # if self.target not in df_raw.columns:
        #     raise ValueError(
        #         f'{self.data_path} 中找不到目标列 {self.target!r}；'
        #         f'现有列：{list(df_raw.columns)}'
        #     )
        cols = [c for c in df_raw.columns if c != self.target]
        self.data_x = df_raw[cols].values
        self.data_y = df_raw[[self.target]].values

        num_rows = len(df_raw)
        num_train = int(num_rows * self.TRAIN_RATIO)
        num_val = int(num_rows * self.VAL_RATIO)

        # 每段必须放得下一整个输入窗口，否则段与段之间会被挤重叠
        # for name, seg in (('val', num_val), ('test', num_rows - num_train - num_val)):
        #     if seg <= self.seq_len:
        #         raise ValueError(
        #             f'{self.data_path}: {name} 段只有 {seg} 行，'
        #             f'不够 seq_len={self.seq_len}，请减小 seq_len 或换更大的文件'
        #         )

        border1s = [self.seq_len - 1, num_train, num_train + num_val]
        border2s = [num_train, num_train + num_val, num_rows]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.label_idx = np.arange(border1, border2)  # 每个样本的标签行号

    def __getitem__(self, index):
        t = int(self.label_idx[index])  # 标签所在时刻

        seq_x = self.data_x[t - self.seq_len + 1: t + 1]
        seq_y = self.data_y[t]

        return seq_x, seq_y

    def __len__(self):
        return len(self.label_idx)


class Dataset_Zero_shot(Dataset):
    """零样本测试集：CSV 的**全部行**都是测试样本，不做 train/val/test 划分。

    用途是把未知工况（如 ``un_state1~2.csv``）整条序列当测试集，评估跨工况泛化。

    - ``flag`` 只接受 ``'test'``，传其它值直接报错（这个类没有训练/验证语义）；
    - 样本定义与 :class:`Dataset_Custom` 一致：
      ``seq_x`` 为最近 ``seq_len`` 行，``seq_y`` 为窗口最后一行；
    - 标签行从 ``seq_len - 1`` 到 ``N - 1``，共 ``N - seq_len + 1`` 个样本，
      整条序列一个不落地全部参与评估。
    """

    def __init__(self, root_path, data_path='ETTh1.csv', flag='test', size=None, target='OT'):
        # size = [seq_len]
        if flag != 'test':
            raise ValueError(f'Dataset_Zero_shot 只接受 flag="test"，收到 {flag!r}')
        # if size is None:
        #     raise ValueError('Dataset_Zero_shot 需要 size=[seq_len]')
        self.seq_len = int(size[0])

        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
        df_raw.columns: [...(process variables), target feature]
        '''
        # if self.target not in df_raw.columns:
        #     raise ValueError(
        #         f'{self.data_path} 中找不到目标列 {self.target!r}；'
        #         f'现有列：{list(df_raw.columns)}'
        #     )
        cols = [c for c in df_raw.columns if c != self.target]
        self.data_x = df_raw[cols].values
        self.data_y = df_raw[[self.target]].values

        num_rows = len(df_raw)
        # if num_rows <= self.seq_len:
        #     raise ValueError(
        #         f'{self.data_path}: 只有 {num_rows} 行，'
        #         f'不够一个 seq_len={self.seq_len} 的输入窗口'
        #     )

        # 全部行都是测试样本：标签行 [seq_len - 1, N)
        self.label_idx = np.arange(self.seq_len - 1, num_rows)

    def __getitem__(self, index):
        t = int(self.label_idx[index])  # 标签所在时刻

        seq_x = self.data_x[t - self.seq_len + 1: t + 1]
        seq_y = self.data_y[t]

        return seq_x, seq_y

    def __len__(self):
        return len(self.label_idx)
