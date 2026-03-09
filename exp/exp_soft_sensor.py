import os
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.cmLoss import cmLoss
import torch
import torch.nn as nn
from torch import optim
import time
import warnings
import numpy as np
import pandas as pd

from transformers import GPT2Tokenizer, GPT2Model

warnings.filterwarnings('ignore')


class Exp_Soft_Sensor(Exp_Basic):
    def __init__(self, args):
        super(Exp_Soft_Sensor, self).__init__(args)
        self.knowledge_base = self._load_kb()
        self.text_emb = self._load_text_emb().to(self.device)
        print("Knowledge Base and Text Embedding initialized.")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args, self.device).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


    def _load_kb(self):
        knowledge_base = []
        for path in self.args.historical_data_path:
            full_path = os.path.join(self.args.root_path, path)
            df = torch.tensor(pd.read_csv(full_path).values).float()
            knowledge_base.append(df.to(self.device))  # 直接移动到设备
        return knowledge_base

    def _load_text_emb(self):
        # 修正路径问题：确保相对于根目录
        model_path = './models/gpt2'
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        # model = GPT2Model.from_pretrained(model_path)

        # === 修改点：将模型移动到计算设备 (GPU) ===
        # model.to(self.device)
        # =======================================

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        with (torch.no_grad()):
            token_ids = tokenizer(
                self.args.text,
                return_tensors="pt",
                padding='max_length',
                max_length=self.args.max_length,
                truncation=True
            ).input_ids.to(self.device)  # 这里数据已经在 GPU 了

            # 现在模型和数据都在 GPU 上，可以正常运行
            if hasattr(self.model, 'module'):
                text_emb = self.model.module.gpt2.get_input_embeddings()(token_ids)
            else:
                text_emb = self.model.gpt2.get_input_embeddings()(token_ids)


        return text_emb

    def _get_data(self, flag, vali_test=False):
        data_set, data_loader = data_provider(self.args, flag, vali_test)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = cmLoss(self.args.task_loss,
                           self.args.similarity_loss,
                           self.args.task_w,
                           self.args.similarity_w,
                           self.args.temperature)
        return criterion

    def _select_vali_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        if hasattr(self.model, 'module'):
            text_emb = self.text_emb.expand(len(self.args.device_ids), -1, -1)
        else:
            text_emb = self.text_emb

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)

        for epoch in range(self.args.train_epochs):
            if hasattr(self.model, 'module'):
                self.model.module.retriever.update_index(self.knowledge_base)
            else:
                self.model.retriever.update_index(self.knowledge_base)
            self.model.train()

            iter_count = 0
            total_losses = []
            task_losses = []
            similarity_losses = []

            epoch_time = time.time()

            for i, (batch_x, batch_y, _, _) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                # loss_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x, text_emb)

                total_loss, task_loss, similarity_loss = criterion(outputs, batch_y)
                total_losses.append(total_loss.item())
                task_losses.append(task_loss.item())
                similarity_losses.append(similarity_loss.item())

                # if (i + 1) % 25 == 0:
                #     # print("top_n_similarity", outputs['top_n_similarity'])
                #     # print("logit_scale", outputs['logit_scale'])
                #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, total_loss.item()))
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()

                total_loss.backward()
                model_optim.step()
                # loss_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            total_losses = np.average(total_losses)
            task_losses = np.average(task_losses)
            similarity_losses = np.average(similarity_losses)

            vali_loss = self.vali(vali_loader, self._select_vali_criterion())
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Task Loss: {3:.7f} Similarity Loss: {4:.7f} Vali Loss: {5:.7f}".format(
                    epoch + 1, train_steps, total_losses, task_losses, similarity_losses, vali_loss))

            if self.args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        return

    def vali(self, vali_loader, criterion):
        total_loss = []
        if hasattr(self.model, 'module'):
            text_emb = self.text_emb.expand(len(self.args.device_ids), -1, -1)
        else:
            text_emb = self.text_emb
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs = self.model(batch_x, text_emb)
                outputs = outputs['output']

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)

        self.model.train()
        total_loss = np.average(total_loss)
        return total_loss

    def test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')
        # knowledge_base = self._load_kb()
        # knowledge_base = [df.to(self.device) for df in knowledge_base]
        if hasattr(self.model, 'module'):
            self.model.module.retriever.update_index(self.knowledge_base)
        else:
            self.model.retriever.update_index(self.knowledge_base)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x, self.text_emb)
                outputs = outputs['output']

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                # break
                # print(pred)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('rmse:{:.4f}, mae:{:.4f}'.format(rmse, mae))
        f = open("result_soft_sensor.txt", 'a')
        f.write(setting + "  \n")
        f.write('rmse:{}, mae:{}'.format(rmse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
