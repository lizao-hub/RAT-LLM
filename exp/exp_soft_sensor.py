"""软测量训练 / 验证 / 测试流程（模型无关）。

模型由 ``run.py --model`` 指定：``models/<model>.py`` 中需要提供 ``Model`` 类，
构造签名为 ``Model(args)``，前向为 ``forward(x) -> y_hat``。

本文件只保留最简单的通用设定（不区分模型类型）：

- 损失函数：``nn.MSELoss``
- 优化器：``torch.optim.Adam``
- 不涉及文本 embedding、检索知识库、相似度蒸馏损失

训练/验证使用已知工况（``--data hlt`` 时按时间 7:2:1 划分）；测试阶段评估两个集合：
已知工况测试集（``flag='test'``）与未知工况全集（``flag='unknown'``），指标为
RMSE / MAE / R²（保留 4 位小数），并写入每个模型一个的 ``result_<model>.txt``。

数据由 ``data_provider`` 提供，每个 batch 为 ``(x, y, _, _)``，``y`` 是下一时刻的目标值。
模型的发现与加载见 ``exp/exp_basic.py``（参考 Time-Series-Library 的 exp_basic 设计）。
"""

import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric, R2
from utils.tools import EarlyStopping

warnings.filterwarnings('ignore')


class Exp_Soft_Sensor(Exp_Basic):
    def __init__(self, args):
        super(Exp_Soft_Sensor, self).__init__(args)

    def _build_model(self):
        # --model 指定 models/ 下的文件名，懒加载后返回 Model 类
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, vali_test=False):
        data_set, data_loader = data_provider(self.args, flag, vali_test)
        seq_x, seq_y, _, _ = data_set[0]
        print(
            f"[{flag}] samples={len(data_set)} | batches={len(data_loader)} | "
            f"x={tuple(seq_x.shape)} y={tuple(seq_y.shape)}"
        )
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=self.args.tmax, eta_min=1e-8
        )

        for epoch in range(self.args.train_epochs):
            self.model.train()

            total_losses = []
            epoch_time = time.time()

            for i, (batch_x, batch_y, _, _) in enumerate(train_loader):
                model_optim.zero_grad(set_to_none=True)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)

                loss.backward()
                model_optim.step()

                total_losses.append(loss.item())

            train_loss = float(np.average(total_losses)) if total_losses else float('nan')
            vali_loss = self.vali(vali_loader, criterion)
            scheduler.step()

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} "
                "Lr: {4:.6f} | cost time: {5:.2f}s".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                    model_optim.param_groups[0]['lr'],
                    time.time() - epoch_time,
                )
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        return

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_x, batch_y, _, _ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred = self.model(batch_x)
                total_loss.append(criterion(pred, batch_y).item())

        self.model.train()
        return float(np.average(total_loss)) if total_loss else float('inf')

    def test(self, setting, test=1):
        # 已知工况测试集：已知工况 7:2:1 划分中的最后 10%
        _, known_loader = self._get_data(flag='test')

        # 未知工况测试集：全部 un_state* 数据；不支持的数据集（如 zero_shot）自动跳过
        try:
            _, unknown_loader = self._get_data(flag='unknown')
        except (AssertionError, ValueError, KeyError) as exc:
            print(f'未知工况集不可用（{exc}），本次只评估已知工况。')
            unknown_loader = None

        if test:
            checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"找不到权重文件 {checkpoint_path}，请先训练或检查 setting 是否一致。"
                )
            print('loading model from', checkpoint_path)
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        known = self._evaluate(known_loader)
        unknown = self._evaluate(unknown_loader)

        self._report(setting, known, unknown)
        return

    # ------------------------------------------------------------ 测试辅助
    def _evaluate(self, loader):
        """在测试集上得到 ``{'preds', 'trues', 'metrics'}``；空集返回 None。"""
        if loader is None or len(loader) == 0:
            return None

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, _, _ in loader:
                batch_x = batch_x.float().to(self.device)
                preds.append(self.model(batch_x).detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        return {'preds': preds, 'trues': trues, 'metrics': self._metrics(preds, trues)}

    @staticmethod
    def _metrics(preds, trues):
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': R2(preds, trues),
            'mse': mse,
            'mape': mape,
            'mspe': mspe,
        }

    def _report(self, setting, known, unknown):
        """打印并按"每个模型一个 result 文件"的要求写入结果（保留 4 位小数）。"""
        for name, result in (('known', known), ('unknown', unknown)):
            if result is None:
                print(f'[{name}] 无可用样本，跳过。')
                continue
            m = result['metrics']
            print(
                '[{:<8}] shape={} rmse={:.4f} mae={:.4f} r2={:.4f}'.format(
                    name, result['preds'].shape, m['rmse'], m['mae'], m['r2']
                )
            )

        # 每个模型一个结果文件：./result_<model>.txt，逐次运行追加
        result_path = './result_{}.txt'.format(self.args.model)
        lines = [setting]
        if known is not None:
            m = known['metrics']
            lines.append(
                '  known  : rmse={:.4f} mae={:.4f} r2={:.4f}'.format(
                    m['rmse'], m['mae'], m['r2']
                )
            )
        if unknown is not None:
            m = unknown['metrics']
            lines.append(
                '  unknown: rmse={:.4f} mae={:.4f} r2={:.4f}'.format(
                    m['rmse'], m['mae'], m['r2']
                )
            )
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n\n')

        # 预测值/指标按 setting 存盘，便于后续画图或复算
        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        for name, result in (('known', known), ('unknown', unknown)):
            if result is None:
                continue
            m = result['metrics']
            np.save(folder_path + f'{name}_pred.npy', result['preds'])
            np.save(folder_path + f'{name}_true.npy', result['trues'])
            np.save(
                folder_path + f'{name}_metrics.npy',
                np.array([m['mae'], m['mse'], m['rmse'], m['r2']]),
            )

        return
