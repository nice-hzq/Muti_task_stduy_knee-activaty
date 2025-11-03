# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ====== 配置 ======
csv_file = r"/home/lenovo/PycharmProjects/CNN+LSTM/data/AB193/Processed/AB193_Circuit_001_post.csv"
save_dir = r"/home/lenovo/PycharmProjects/CNN+LSTM/data/ClipPredict_AB193"
os.makedirs(save_dir, exist_ok=True)

# 选取片段（闭区间左开右闭）
start_idx, end_idx = 1000, 14900   # 想全量就把这两行注释掉并用 df 全部

# 滑窗参数
window_size = 200
stride = 100

# 特征列
imu_columns = [
    'Right_Shank_Ax', 'Right_Shank_Ay', 'Right_Shank_Az',
    'Right_Shank_Gx', 'Right_Shank_Gy', 'Right_Shank_Gz',
    'Right_Thigh_Ax', 'Right_Thigh_Ay', 'Right_Thigh_Az',
    'Right_Thigh_Gx', 'Right_Thigh_Gy', 'Right_Thigh_Gz',
    'Left_Shank_Ax', 'Left_Shank_Ay', 'Left_Shank_Az',
    'Left_Shank_Gx', 'Left_Shank_Gy', 'Left_Shank_Gz',
    'Left_Thigh_Ax', 'Left_Thigh_Ay', 'Left_Thigh_Az',
    'Left_Thigh_Gx', 'Left_Thigh_Gy', 'Left_Thigh_Gz',
    'Waist_Ax', 'Waist_Ay', 'Waist_Az',
    'Waist_Gx', 'Waist_Gy', 'Waist_Gz',
]
emg_columns = [
    'Right_TA', 'Right_MG', 'Right_SOL', 'Right_BF', 'Right_ST', 'Right_VL', 'Right_RF',
    'Left_TA', 'Left_MG', 'Left_SOL', 'Left_BF', 'Left_ST', 'Left_VL', 'Left_RF'
]

# 训练里你现在用的是纯 IMU；如果预测也只想喂 IMU，就保持如下：
# input_columns = imu_columns
# 如果需要 IMU+EMG（共 44 通道），改成：
input_columns = imu_columns + emg_columns

# 标签列
label_cls_col = 'Mode'
label_reg_cols = ['Right_Knee', 'Left_Knee']


def prepare_clip_for_infer(csv_path: str):
    df = pd.read_csv(csv_path)

    # 选段
    if 'start_idx' in globals() and 'end_idx' in globals() and start_idx is not None and end_idx is not None:
        df = df.iloc[start_idx:end_idx].reset_index(drop=True)

    # ---- EMG 仅做 Z-score（和你训练处理一致）----
    # 即便 input_columns 不含 EMG，也做一下标准化，保持流程一致（不影响最终输入）
    exist_emg = [c for c in emg_columns if c in df.columns]
    if exist_emg:
        scaler = StandardScaler()
        df.loc[:, exist_emg] = scaler.fit_transform(df[exist_emg])
    else:
        print("⚠️ 未找到 EMG 列（将跳过 EMG 标准化）")

    # 取输入与标签
    # 若有列缺失，会报错，便于尽早发现
    X_mat = df[input_columns].to_numpy()
    mode_vec = df[label_cls_col].to_numpy()
    rk = df[label_reg_cols[0]].to_numpy()
    lk = df[label_reg_cols[1]].to_numpy()

    # 保证分类标签是非负整数（bincount 需要）
    if not np.issubdtype(mode_vec.dtype, np.integer):
        # 如果是字符串类别，可自行映射；这里先尝试 astype 失败则报错
        try:
            mode_vec = mode_vec.astype(np.int64)
        except Exception as e:
            raise ValueError(f"{label_cls_col} 不是整数标签，请先映射为整数。原始 dtype={mode_vec.dtype}, 示例值={mode_vec[:5]}") from e

    # 滑窗
    X, y_cls, y_reg = [], [], []
    L = len(df)
    for i in range(0, L - window_size + 1, stride):
        segX = X_mat[i:i+window_size]
        seg_mode = mode_vec[i:i+window_size]
        seg_r = rk[i:i+window_size]
        seg_l = lk[i:i+window_size]

        X.append(segX)
        # 活动标签取众数
        y_cls.append(np.bincount(seg_mode).argmax())
        # 回归标签取均值
        y_reg.append([seg_r.mean(), seg_l.mean()])

    X = np.asarray(X, dtype=np.float32)          # (N,T,C)
    y_cls = np.asarray(y_cls, dtype=np.int64)    # (N,)
    y_reg = np.asarray(y_reg, dtype=np.float32)  # (N,2)
    return X, y_cls, y_reg


if __name__ == "__main__":
    X, y_cls, y_reg = prepare_clip_for_infer(csv_file)

    np.save(os.path.join(save_dir, "X_windows.npy"), X)
    np.save(os.path.join(save_dir, "y_cls.npy"), y_cls)
    np.save(os.path.join(save_dir, "y_reg.npy"), y_reg)

    print(f"✅ 预测片段滑窗已保存：")
    print(f"   X_windows: {X.shape} (N,T,C)")
    print(f"   y_cls:     {y_cls.shape} (N,)")
    print(f"   y_reg:     {y_reg.shape} (N,2)")
    print(f"📁 保存目录：{save_dir}")
