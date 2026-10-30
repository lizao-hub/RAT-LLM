"""零样本测试流程：只做测试，不训练。

把未知工况（如 ``un_state1~2.csv``）的**全部行**当作测试集，加载训练阶段保存的
checkpoint，评估跨工况泛化能力。数据侧由 ``data_provider`` 的
``Dataset_Zero_shot`` 提供：``x`` 为最近 ``seq_len`` 行，``y`` 为窗口最后一行。

用法（``--data zero_shot`` + ``--task_name zero_shot``，``--is_training 0``）::

    python run.py --task_name zero_shot --is_training 0 \
        --model LSTM --model_id hlt_lstm_lr0.001_dm64_el1 \
        --data zero_shot --root_path ./datasets/hlt \
        --data_path un_state1.csv un_state2.csv \
        --target 11 --enc_in 26 --seq_len 48 \
        --batch_size 512 --learning_rate 0.001 \
        --d_model 64 --e_layers 1 --gpu 0

注意：checkpoint 路径是 ``./checkpoints/<setting>/checkpoint.pth``，而 ``setting``
由 ``run.py`` 用 ``model_id / model / seq_len / patch_size / gpt_layers / top_n /
temperature / learning_rate / batch_size`` 拼成，所以**这些参数必须和训练时完全一致**，
否则会找不到权重（这里会直接抛 ``FileNotFoundError`` 而不是静默用随机初始化）。

输出（不覆盖训练阶段 known-condition 的结果）::

    ./result_zero_shot.txt                        # 逐组追加的指标
    ./results/<setting>_zero_shot/unknown_pred.npy
    ./results/<setting>_zero_shot/unknown_true.npy
    ./results/<setting>_zero_shot/unknown_metrics.npy   # [rmse, mae, r2]

文件名沿用 ``scripts/visualize_results.py`` 里 ``unknown_*`` 的约定。
"""

import os
import warnings

import numpy as np
import torch
import torch.nn as nn

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric

warnings.filterwarnings('ignore')


class Exp_Zero_Shot(Exp_Basic):
    def __init__(self, args):
        super(Exp_Zero_Shot, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag='test'):
        data_set, data_loader = data_provider(self.args, flag)
        seq_x, seq_y = data_set[0]
        print(
            f'[{flag}] samples={len(data_set)} | batches={len(data_loader)} | '
            f'x={tuple(seq_x.shape)} y={tuple(seq_y.shape)}'
        )
        return data_set, data_loader

    def test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')

        ckpt_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if test:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f'找不到训练好的权重：{ckpt_path}\n'
                    'setting 由 model_id/model/seq_len/... 拼成，'
                    '零样本测试的 model_id、seq_len、learning_rate、batch_size 等'
                    '必须和训练时一致。'
                )
            print(f'loading model: {ckpt_path}')
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('zero-shot test shape:', preds.shape, trues.shape)

        folder_path = os.path.join('./results', setting + '_zero_shot')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        rmse, mae, r2 = metric(preds, trues)
        print('rmse:{}, mae:{}, r2:{}'.format(rmse, mae, r2))
        f = open('result_zero_shot.txt', 'a')
        f.write(setting + '  \n')
        f.write('rmse:{}, mae:{}, r2:{}'.format(rmse, mae, r2))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(os.path.join(folder_path, 'unknown_metrics.npy'), np.array([rmse, mae, r2]))
        np.save(os.path.join(folder_path, 'unknown_pred.npy'), preds)
        np.save(os.path.join(folder_path, 'unknown_true.npy'), trues)

        return
