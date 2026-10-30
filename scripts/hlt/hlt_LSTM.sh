#!/usr/bin/env bash
# =============================================================================
# LSTM 基线 · HLT 数据集 · 网格搜索（Linux 服务器版）
# -----------------------------------------------------------------------------
# 搜索空间：
#   learning_rate : 0.005  0.001  0.01
#   d_model       : 32  64  128  256      （LSTM 隐层维度）
#   e_layers      : 1  2                  （LSTM 层数）
#   共 3 × 4 × 2 = 24 组
#
# 训练：已知工况 state1~3.csv 按时间 7:1:2 切 train/val/test（不打乱）；
#       划分在 data_provider/data_loader.py 里按“标签行”切，三段零重叠
# 测试：已知工况测试集（最后 20%）；未知工况 un_state1~2.csv 暂未接入，
#       不要放进 --data_path，否则会被当训练数据切分
# 权重：./checkpoints/<setting>/checkpoint.pth
# 预测：./results/<setting>/{pred,true,metrics}.npy
# 指标：./result_soft_sensor.txt（逐组追加，按 setting 区分）
#
# model_id 里带上了 lr / d_model / e_layers（run.py 的 setting 只含 lr 和
# batch_size 等，不含 d_model / e_layers），否则不同组合会互相覆盖权重和结果：
#   hlt_lstm_lr<lr>_dm<d_model>_el<e_layers>
#
# 用法（服务器上任意目录执行都可以）：
#   conda activate ratllm
#   bash scripts/HLT/hlt_LSTM.sh
#
# 不激活环境：
#   PYTHON=/data/anonym5/anaconda5/envs/ratllm/bin/python bash scripts/HLT/hlt_LSTM.sh
#
# 可选环境变量：PYTHON / GPU / TRAIN_EPOCHS / BATCH_SIZE / SEQ_LEN / NUM_WORKERS
#   GPU=0 bash scripts/HLT/hlt_LSTM.sh          # 指定物理卡
#   TRAIN_EPOCHS=5 GPU=0 bash scripts/HLT/hlt_LSTM.sh   # 先小 epoch 粗筛
# =============================================================================

set -euo pipefail

# 脚本位于 <project>/scripts/HLT/ 下：先定位工程根目录，保证 ./datasets、./checkpoints
# 等相对路径与从工程根直接运行时一致
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# 解释器：默认取 PATH 里的 python（即 conda activate 后的环境）
PYTHON="${PYTHON:-python}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "[ERROR] 找不到解释器：${PYTHON}" >&2
    echo "        请先 conda activate <env>，或用 PYTHON=/path/to/python 指定绝对路径。" >&2
    exit 1
fi

# 实时刷新日志，避免 nohup 下看不到进度
export PYTHONUNBUFFERED=1

# =============================================================================
# 网格搜索空间
# =============================================================================
learning_rate_list=(0.005 0.001 0.01)
d_model_list=(32 64 128 256)
e_layers_list=(1 2)

# 想缩小范围就注释掉不想要的取值，例如只跑 64/128：
# learning_rate_list=(0.001)
# d_model_list=(64 128)
# e_layers_list=(1 2)

# =============================================================================
# 固定超参
# =============================================================================
seq_len="${SEQ_LEN:-48}"   # 输入窗口长度：48（x 为最近 48 个采样点，标签取窗口最后一步）
batch_size="${BATCH_SIZE:-512}"
train_epochs="${TRAIN_EPOCHS:-50}"
patience=10
num_workers="${NUM_WORKERS:-0}"
dropout=0.3

# 单卡：--gpu 指定物理卡号（0-based），多卡见文件末尾注释
gpu="${GPU:-0}"

# =============================================================================
# 数据 / 模型设置
# =============================================================================
model=LSTM
data=custom
root_path="./datasets/hlt"   # 需含 state1~3.csv
# 只放已知工况；un_state1~2.csv 属于未知工况，留给零样本测试用，不要加进来
data_path_list=(state1.csv state2.csv state3.csv)
target=11                    # 质量变量列名（第 27 列，列名为 '11'）
enc_in=26                    # 26 个工艺变量输入

# =============================================================================
# 数据自检：缺文件时直接给出明确提示，而不是等到 dataloader 里抛异常
# =============================================================================
for f in "${data_path_list[@]}"; do
    if [ ! -f "${root_path}/${f}" ]; then
        echo "[ERROR] 缺少数据文件：${PROJECT_ROOT}/${root_path#./}/${f}" >&2
        exit 1
    fi
done

# =============================================================================
# 网格搜索执行
# =============================================================================
total_experiments=$(( ${#learning_rate_list[@]} * ${#d_model_list[@]} * ${#e_layers_list[@]} ))
index=0
failed=()

echo "======================================================="
echo "LSTM Grid Search on HLT Dataset"
echo "======================================================="
echo "  python       : $(command -v "${PYTHON}")"
echo "  project root : ${PROJECT_ROOT}"
echo "  root_path    : ${root_path}"
echo "  data_path    : ${data_path_list[*]}"
echo "  data / target: ${data} / ${target} (enc_in=${enc_in})"
echo "  seq_len      : ${seq_len}"
echo "  batch_size   : ${batch_size}"
echo "  train_epochs : ${train_epochs}"
echo "  patience     : ${patience}"
echo "  num_workers  : ${num_workers}"
echo "  dropout      : ${dropout}"
echo "  gpu          : ${gpu}"
echo ""
echo "Search space:"
echo "  learning_rate : ${learning_rate_list[*]}"
echo "  d_model       : ${d_model_list[*]}"
echo "  e_layers      : ${e_layers_list[*]}"
echo "  total         : ${total_experiments} combinations"
echo "======================================================="
echo ""

for learning_rate in "${learning_rate_list[@]}"; do
    for d_model in "${d_model_list[@]}"; do
        for e_layers in "${e_layers_list[@]}"; do

            index=$((index + 1))
            model_id="hlt_lstm_lr${learning_rate}_dm${d_model}_el${e_layers}"

            echo ""
            echo "[Experiment ${index}/${total_experiments}]"
            echo "  learning_rate=${learning_rate}"
            echo "  d_model=${d_model}"
            echo "  e_layers=${e_layers}"
            echo "  model_id=${model_id}"
            echo ""

            cmd_args=(
                --task_name soft_sensor
                --is_training 1
                --model "${model}"
                --model_id "${model_id}"
                --data "${data}"
                --root_path "${root_path}"
                --data_path "${data_path_list[@]}"
                --target "${target}"
                --enc_in "${enc_in}"
                --seq_len "${seq_len}"
                --batch_size "${batch_size}"
                --learning_rate "${learning_rate}"
                --train_epochs "${train_epochs}"
                --patience "${patience}"
                --num_workers "${num_workers}"
                --dropout "${dropout}"
                --d_model "${d_model}"
                --e_layers "${e_layers}"
                --gpu "${gpu}"
                --use_gpu 1
            )

            # 单组失败不中断整个网格（例如某一组显存不够），最后统一汇总
            if "${PYTHON}" run.py "${cmd_args[@]}"; then
                echo "[DONE ${index}/${total_experiments}] ${model_id}"
            else
                echo "[FAIL ${index}/${total_experiments}] ${model_id}" >&2
                failed+=("${model_id}")
            fi

        done
    done
done

echo ""
echo "======================================================="
echo "Grid Search Completed! Total experiments: ${index}"
echo "Results: ./result_soft_sensor.txt  (per setting / per run)"
echo "Weights: ./checkpoints/<setting>/checkpoint.pth"
echo "Preds  : ./results/<setting>/"
echo "======================================================="
if [ ${#failed[@]} -gt 0 ]; then
    echo "失败的组合（${#failed[@]} 个）：${failed[*]}" >&2
    exit 1
fi

# 后台跑（服务器断开 SSH 也不中断）：
#   nohup bash scripts/HLT/hlt_LSTM.sh > hlt_LSTM_grid.log 2>&1 &
#
# 多卡（如需）：把 cmd_args 里的 --gpu "${gpu}" --use_gpu 1 换成
#   --use_multi_gpu --devices "0,1" --use_gpu 1
#
# 无 GPU 或想强制 CPU：直接 CUDA_VISIBLE_DEVICES="" bash scripts/HLT/hlt_LSTM.sh，
# exp_basic 会自动回落到 CPU（24 组会很慢，仅用于冒烟）
