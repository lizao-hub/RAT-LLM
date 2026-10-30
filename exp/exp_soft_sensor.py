"""软测量训练 / 验证 / 测试流程（模型无关）。

模型由 ``run.py --model`` 指定：``models/<model>.py`` 中需提供 ``Model`` 类，
构造签名为 ``Model(args)``，前向为 ``forward(x) -> y_hat``。

- 数据：每个 batch 为 ``(x, y)``，``x`` 形状 ``[B, seq_len, C]``，
  ``y`` 是输入窗口最后一个时刻的目标值（单点回归）；
- 损失 / 优化器：``nn.MSELoss`` + ``torch.optim.Adam``；
- 学习率：余弦调度 ``lr = base_lr / 2 * (1 + cos(epoch / train_epochs * pi))``，
  每个 epoch 开始时按当前 epoch 重设（首个 epoch 为基础学习率，末个 epoch 衰减到 ≈0）；
- 测试指标：RMSE / MAE / R²，追加写入 ``result_soft_sensor.txt``。

模型的发现与加载见 ``exp/exp_basic.py``。
"""

import math
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping

warnings.filterwarnings('ignore')


class Exp_Soft_Sensor(Exp_Basic):
    def __init__(self, args):
        super(Exp_Soft_Sensor, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        seq_x, seq_y = data_set[0]
        print(
            f'[{flag}] samples={len(data_set)} | batches={len(data_loader)} | '
            f'x={tuple(seq_x.shape)} y={tuple(seq_y.shape)}'
        )
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            # 余弦调度：base_lr -> 0，每个 epoch 开始时按当前 epoch 重设
            lr = self.args.learning_rate / 2 * (1 + math.cos(epoch / self.args.train_epochs * math.pi))
            for param_group in model_optim.param_groups:
                param_group['lr'] = lr

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} | Lr: {5:.6f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss,
                model_optim.param_groups[0]['lr']))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'),
                           map_location=self.device)
            )

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = os.path.join('./results', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        rmse, mae, r2 = metric(preds, trues)
        print('rmse:{}, mae:{}, r2:{}'.format(rmse, mae, r2))
        f = open("result_soft_sensor.txt", 'a')
        f.write(setting + "  \n")
        f.write('rmse:{}, mae:{}, r2:{}'.format(rmse, mae, r2))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + '/metrics.npy', np.array([rmse, mae, r2]))
        np.save(folder_path + '/pred.npy', preds)
        np.save(folder_path + '/true.npy', trues)

        return
