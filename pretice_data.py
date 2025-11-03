import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from model.model import CNNLSTM_MultiTask
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def load_model_and_predict(model, model_path, input_seq, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    加载模型并对输入数据进行预测
    - model: 初始化后的模型结构
    - model_path: 最佳模型的 .pth 文件路径
    - input_seq: 输入张量，形状 (T, seq_len, num_features)
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    input_seq = input_seq.to(device)  # [T, 25, 44]
    with torch.no_grad():
        out_cls, out_right, out_left = model(input_seq)

    return out_right.cpu().numpy(), out_left.cpu().numpy()


def plot_joint_angle_predictions(gt_right, gt_left, pred_right, pred_left, save_path=None):
    """
    绘制预测曲线与真实角度
    """
    t = np.arange(len(gt_right))
    plt.figure(figsize=(12, 6))

    # 右膝
    plt.subplot(2, 1, 1)
    plt.plot(t, gt_right, label='Ground Truth - Right', linestyle='--')
    plt.plot(t, pred_right, label='Prediction - Right')
    plt.title("Right Knee Angle Prediction")
    plt.ylabel("Angle (°)")
    plt.legend()
    plt.grid(True)

    # 左膝
    plt.subplot(2, 1, 2)
    plt.plot(t, gt_left, label='Ground Truth - Left', linestyle='--')
    plt.plot(t, pred_left, label='Prediction - Left')
    plt.title("Left Knee Angle Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("Angle (°)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()


# ==== 准备测试输入 ====
# 假设你手动加载某段 CSV 或 npy 数据作为 input_seq
# 例如：input_seq = torch.from_numpy(my_clip).unsqueeze(0).float()  # shape: [1, 25, 44]
# 或多个片段：[T, 25, 44]

# ==== 加载剪辑数据 ====
clip_dir = r"D:/exercise/CNN+LSTM/data/ClipPredict"  # 你刚刚保存的片段路径
X = np.load(os.path.join(clip_dir, "X_windows.npy"))      # (N, 25, 44)
y_r = np.load(os.path.join(clip_dir, "y_right.npy"))      # (N,)
y_l = np.load(os.path.join(clip_dir, "y_left.npy"))       # (N,)

# ==== 数据转换 ====
input_seq = torch.tensor(X, dtype=torch.float32)          # 模型输入
ground_truth = np.stack([y_r, y_l], axis=1)               # 拼接为 (N, 2)

# ==== 模型加载 ====
model = CNN_MultiTask(input_size=44, num_classes=7)    # 替换为你的参数
model_path = "D:\exercise\CNN+LSTM\logs\\run_20250802_165523\\best_model.pth"

pred_right, pred_left = load_model_and_predict(model, model_path, input_seq)

# ==== 可视化 ====
gt_right = ground_truth[:, 0]
gt_left = ground_truth[:, 1]

save_fig = os.path.join("D:\exercise\CNN+LSTM\logs\\run_20250802_165523", "tracking_plot.png")
plot_joint_angle_predictions(gt_right, gt_left, pred_right, pred_left, save_path=save_fig)

print("✅ 跟踪曲线图已保存：", save_fig)