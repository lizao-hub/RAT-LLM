#!/usr/bin/env bash
# 在 hlt 数据集（3 个已知工况 state1~3 + 2 个未知工况 un_state1~2）上训练 LSTM 基线。
# 训练：已知工况按时间 7:2:1 切分 train/val/test（不打乱）
# 测试：已知工况测试集（最后 10%）+ 全部未知工况，指标写入 ./result_LSTM.txt
# 权重：./checkpoints/<setting>/checkpoint.pth
# 预测：./results/<setting>/{known,unknown}_{pred,true,metrics}.npy
# 画图：python scripts/visualize_results.py --model LSTM

# 解释器：优先用环境变量 PYTHON，否则用 conda pytorch 环境的绝对路径
# （Windows 上 git bash 看不到 conda activate 的环境，必须显式指定）
PYTHON=${PYTHON:-"D:/Software/Anaconda3/envs/pytorch/python.exe"}

seq_len=48
#label_len=48
#pred_len=1
batch_size=64
lr=0.001
train_epochs=50
patience=10
num_workers=0
gpu=0

"$PYTHON" run.py \
    --task_name soft_sensor \
    --is_training 1 \
    --model LSTM \
    --model_id hlt \
    --data zero_shot \
    --root_path ./datasets/hlt \
    --target 11 \
    --enc_in 26 \
    --seq_len $seq_len \
    --batch_size $batch_size \
    --learning_rate $lr \
    --train_epochs $train_epochs \
    --patience $patience \
    --num_workers $num_workers \
    --gpu $gpu \
    --use_gpu 1

# 多卡（如需）：把最后一行换成 --use_multi_gpu --devices 0,1
