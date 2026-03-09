import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# 1. 完善模型路径与名称
models_info = {
    # '1CALF': 'results/PD23/CALF_ytt_CALF_48_3_0.0002_256/',
    # '1GPT4TS': 'results/PD23/GPT4TS_ytt_GPT4TS_48_4_4_256_6_4_1_0.001_256/',
    # '1iTransformer': 'results/PD23/ytt_itr_iTransformer_48_10_10_128_3_4_1_0.00015_256/',
    # '1Crossformer': 'results/PD23/ytt_c_Crossformer_48_10_10_1024_3_4_1_0.00015_256/',
    # '1LSTM': 'results/PD23/ytt_lstm4_LSTM_48_10_10_256_3_4_1_0.00025_1024/',
    '1RA4SS': 'results/PD23/ytt4_RASS_5_48_6_6_768_6_4_1_0.0005_1024/',
    # '1Bi-LSTM': 'results/PD23/ytt_lstm4_Bi_LSTM_48_10_10_256_3_4_1_0.00025_1024/',
    # '1STA-LSTM': 'results/PD23/ytt4_lstm_STA_LSTM_48_10_10_256_3_4_1_0.00075_256/',
    # '1TCN': 'results/PD23/ytt4_TCN_TCN_48_10_10_256_3_4_1_0.0005_256/',
    '1': 'results/PD23/ytt_random_ret_RASS_5_random_ret_48_4_4_768_6_4_0.5_0.001_1024/',
    '2': 'results/PD23/ytt4_wo_rand_wins_RASS_5_48_6_6_768_6_4_1_0.0005_1024/',
    '3': 'results/PD23/ytt4_wo_rib_RASS_5_wo_text/',
    '4': 'results/PD23/ytt4_RASS_5_wo_all_RIB_48_6_6_768_6_4_1_0.0005_1024/',
}


def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


# 设置 TII 顶刊绘图风格
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.unicode_minus'] = False

# 创建保存目录
output_dir = 'pd_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"{'Model':<18} | {'Mode':<10} | {'RMSE':<8} | {'MAE':<8} | {'R2':<8}")
print("-" * 70)

split_idx = 1000

# 3. 循环处理数据并独立绘图
for i, (name, path) in enumerate(models_info.items()):
    try:
        # 加载数据
        pred_raw = np.load(path + 'pred.npy').flatten()
        true_raw = np.load(path + 'true.npy').flatten()

        # 数据切片
        # t_k, p_k = true_raw[1800:2800], pred_raw[700:1700]
        # t_u, p_u = true_raw[6300:8300], pred_raw[-3000:-1000]
        t_k, p_k = true_raw[1750:2750], pred_raw[1750:2750]
        t_u, p_u = true_raw[6300:8300], pred_raw[6300:8300]

        y_true = np.concatenate([t_k, t_u])
        y_pred = np.concatenate([p_k, p_u])
        total_len = len(y_true)

        # 打印指标
        m_k = calculate_metrics(t_k, p_k)
        m_u = calculate_metrics(t_u, p_u)
        print(f"{name:<18} | Known      | {m_k[0]:.4f} | {m_k[1]:.4f} | {m_k[2]:.4f}")
        print(f"{'':<18} | Unknown    | {m_u[0]:.4f} | {m_u[1]:.4f} | {m_u[2]:.4f}")

        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        # 绘制曲线
        ax.plot(y_true, label="True", color="#1f77b4", linewidth=1.5, zorder=2)
        ax.plot(y_pred, label="Pred", color="#d62728", linewidth=1.5, zorder=2, linestyle='--')

        mean_k = np.mean(t_k)
        mean_u = np.mean(t_u)

        # 绘制已知工况均值线 (0 到 split_idx)
        ax.hlines(mean_k, 0, split_idx, colors='gray', linestyles=':', linewidth=2, zorder=2, label="Mean")
        # 绘制未知工况均值线 (split_idx 到 total_len)
        ax.hlines(mean_u, split_idx, total_len, colors='gray', linestyles=':', linewidth=2, zorder=2)

        # 计算分割点
        split_idx = len(t_k)
        total_len = len(y_true)

        # 1. 绘制分割线：改为虚线 (linestyle='--')
        ax.axvline(x=split_idx, color='black', linestyle='--', linewidth=1.5, zorder=4)

        # 2. 将文字标注移动到图表内部，横线（顶部边缘）下方
        # y=0.95 表示在绘图区域高度的 95% 处，va='top' 确保文字向下排列
        ax.text(split_idx / 2, 0.95, 'Known Mode',
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=20, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))  # 添加半透明背景防止被曲线遮挡

        ax.text(split_idx + (total_len - split_idx) / 2, 0.95, 'Unknown Mode',
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=20, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        # 3. 坐标轴细节优化 (TII 风格：四周封闭且刻度向内)
        ax.tick_params(direction='in', top=True, right=True, labelsize=12)
        ax.set_xlabel('Sample Index', fontsize=20)
        ax.set_ylabel('Value', fontsize=20)

        # 设置 x 轴范围，防止左右留白
        ax.set_xlim(0, total_len)

        # 4. 图例优化：放在右上角，边框设为黑色以符合期刊要求
        ax.legend(loc='upper right', fontsize=12, frameon=True, edgecolor='black', fancybox=False)

        # 调整布局
        plt.tight_layout()

        # 保存图像 (支持 PDF 和 PNG)
        file_name = f"{output_dir}/Comparison_{name}.pdf"
        # plt.savefig(file_name, bbox_inches='tight')
        # plt.show() # 如果不需要在窗口连续弹出，可以注释掉
        plt.close(fig)  # 释放内存，防止内存泄漏

    except FileNotFoundError:
        print(f"Skipping {name}: File not found at {path}")

print(f"\nAll plots have been saved to the '{output_dir}' folder.")