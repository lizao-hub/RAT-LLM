import argparse
import os
from exp.exp_soft_sensor import Exp_Soft_Sensor
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAG')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast', help='task name')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, help='model name')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--test_data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--train_data_path', type=str, nargs='+', default=['ETTh1.csv'], help='list of train data files')
    parser.add_argument('--historical_data_path', type=str, nargs='+', default=['ETTh1.csv'], help='list of historical data files')

    parser.add_argument('--text', type=str, default='Zero-shot soft sensing')
    parser.add_argument('--max_length', type=int, default=50)

    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')

    # model define
    parser.add_argument('--patch_size', type=int, default=10)
    parser.add_argument('--patch_size_list', type=int, nargs='+', default=[16, 8], help='list of patch size')
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--stride_ret', type=int, default=32)
    parser.add_argument('--top_k', type=int, default=2, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=768, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=6, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--d_hid', type=int, default=128, help='dimension of hid')
    parser.add_argument('--d_cib', type=int, default=32, help='dimension of Retriever')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')

    # rag
    parser.add_argument('--top_n', type=int, default=4, help='top_n')
    parser.add_argument('--temperature', type=float, default=1)

    # lora
    parser.add_argument('--r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--similarity_loss', type=str, default='kl', help='distillation loss function')
    parser.add_argument('--task_loss', type=str, default='mse', help='task loss function')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='Use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    # self
    parser.add_argument('--tmax', type=int, default=10)
    parser.add_argument('--cos', type=int, default=1)

    # loss weight
    parser.add_argument('--task_w', type=float, default=1.0)
    parser.add_argument('--similarity_w', type=float, default=0.5)

    # gpt
    parser.add_argument('--gpt_layers', type=int, default=3, help='number of hidden layers in gpt')
    parser.add_argument('--word_embedding_path', type=str, default="wte_pca_500.pt")
    parser.add_argument('--last_prompt_token_path', type=str, default="last_prompt_token.pt")

    args = parser.parse_args()

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    import torch

    def set_seed(seed):
        random.seed(seed)  # Python 随机模块
        np.random.seed(seed)  # NumPy
        torch.manual_seed(seed)  # CPU
        torch.cuda.manual_seed_all(seed)  # 所有 GPU
        torch.backends.cudnn.deterministic = True  # 确保 CUDNN 使用确定性算法
        torch.backends.cudnn.benchmark = False  # 关闭 CUDNN 自动优化
        # torch.use_deterministic_algorithms(True)  # PyTorch 1.8+ 支持
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # CUDA 算法配置
        os.environ['PYTHONHASHSEED'] = str(seed)  # Python 哈希随机化

    fix_seed = 2026
    set_seed(fix_seed)

    if args.task_name == 'soft_sensor':
        Exp = Exp_Soft_Sensor

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.seq_len,
                args.patch_size,
                args.stride,
                args.d_ff,
                args.gpt_layers,
                args.top_n,
                args.temperature,
                args.learning_rate,
                args.batch_size,
                )

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.seq_len,
            args.patch_size,
            args.stride,
            args.d_ff,
            args.gpt_layers,
            args.top_n,
            args.temperature,
            args.learning_rate,
            args.batch_size,
        )

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
