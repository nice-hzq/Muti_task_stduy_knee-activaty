# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# ========= 改这里（你的实际路径）=========
CKPT_PATH = r"/home/lenovo/PycharmProjects/CNN+LSTM/data/AB193/logs/run_20250810_184821/fold_6/best_model.pth"
SAVE_DIR  = r"/home/lenovo/PycharmProjects/CNN+LSTM/data/ClipPredict_AB193"
X_PATH    = os.path.join(SAVE_DIR, "X_windows.npy")   # (N,T,44)
Y_REG_PATH= os.path.join(SAVE_DIR, "y_reg.npy")       # (N,2) -> [:,0]=Right, [:,1]=Left
Y_CLS_PATH= os.path.join(SAVE_DIR, "y_cls.npy")       # (N,)
FIG_PATH  = os.path.join(SAVE_DIR, "tracking_with_activity.png")
INPUT_SIZE = 44                                       # 与训练一致
# ========================================
# ====== 活动背景着色配置（7类：0~6）======
USE_ACTIVITY_BG   = True    # 是否在角度子图叠加活动背景
SHOW_PRED_BG      = True    # 是否叠加“预测活动”的背景
USE_GRAYSCALE     = False   # 论文黑白打印可设 True（预测层用斜纹区分）

# 活动ID -> 名称（0~6）
CLASS_NAMES = {
    0: "Sitting",      # sit
    1: "Level ground walking",      # walk
    2: "Ramp ascent",
    3: "Ramp descent",
    4: "Stair ascent",
    5: "Stair descent",
    6: "Standing"       # stand
}

# 每个类别的颜色（彩色展示时更清晰；黑白时会自动转灰度+斜纹）
CLASS_COLORS = {
    0: "#1f77b4",  # 坐
    1: "#ff7f0e",  # 走
    2: "#2ca02c",  # 上坡
    3: "#d62728",  # 下坡
    4: "#9467bd",  # 上楼梯
    5: "#8c564b",  # 下楼梯
    6: "#17becf",  # 站
}

def _segments_from_labels(labels: np.ndarray):
    """把离散标签序列分成 [(start, end, cls_id), ...] 区间，end 为包含端"""
    segs = []
    if len(labels) == 0: return segs
    s, cur = 0, int(labels[0])
    for i in range(1, len(labels)):
        if int(labels[i]) != cur:
            segs.append((s, i-1, cur))
            s, cur = i, int(labels[i])
    segs.append((s, len(labels)-1, cur))
    return segs

def _color_for_class(cid: int, idx: int = 0):
    """给类别挑颜色并设置透明度；idx=0 表示GT层，idx=1 表示Pred层"""
    import matplotlib.colors as mcolors
    base = CLASS_COLORS.get(int(cid), list(plt.cm.tab10.colors)[int(cid) % 10])
    alpha = 0.18 if idx == 0 else 0.10  # Pred 更浅
    return mcolors.to_rgba(base, alpha=alpha)

from model.model import CNNLSTM_MultiTask  # 按你的工程结构

def _load_X_windows(x_path):
    # 兼容 numpy/tensor 保存
    try:
        X = np.load(x_path)
        if isinstance(X, np.ndarray) and X.dtype != object:
            return X.astype(np.float32)
        X = np.load(x_path, allow_pickle=True)
        if hasattr(X, "item"):
            X = X.item()
    except Exception:
        X = np.load(x_path, allow_pickle=True)
    if hasattr(X, "numpy"):
        return X.numpy().astype(np.float32)
    if isinstance(X, np.ndarray) and X.dtype == object:
        if X.shape == ():
            inner = X.item()
            if hasattr(inner, "numpy"):
                return inner.numpy().astype(np.float32)
        X = np.stack([t.numpy() if hasattr(t, "numpy") else np.asarray(t) for t in X], axis=0)
        return X.astype(np.float32)
    raise RuntimeError(f"未能识别 {x_path} 的内容格式，请保存为 numpy 数组。")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Info] device:", device)

    # 1) 读取数据
    X      = _load_X_windows(X_PATH)                    # (N,T,44)
    y_reg  = np.load(Y_REG_PATH).astype(np.float32)     # (N,2)
    y_cls  = np.load(Y_CLS_PATH).astype(np.int64)       # (N,)
    y_r, y_l = y_reg[:, 0], y_reg[:, 1]

    if X.ndim != 3: raise ValueError(f"X 应为 (N,T,C)，当前 {X.shape}")
    N, T, C = X.shape
    assert len(y_r)==len(y_l)==len(y_cls)==N, "X 与 y 长度不一致"
    if C != INPUT_SIZE: raise ValueError(f"C={C} 与 INPUT_SIZE={INPUT_SIZE} 不一致")

    print(f"[Info] X:{X.shape}, y_reg:{y_reg.shape}, y_cls:{y_cls.shape}")

    # 2) 模型 & 权重
    model = CNNLSTM_MultiTask(input_size=INPUT_SIZE).to(device).eval()
    state = torch.load(CKPT_PATH, map_location="cpu")
    if isinstance(state, dict) and any(k in state for k in ["state_dict","model"]):
        state = state.get("state_dict", state.get("model"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:   print("[Warn] missing keys:", missing)
    if unexpected:print("[Warn] unexpected keys:", unexpected)

    # 3) 推理
    preds_r, preds_l, preds_cls = [], [], []
    with torch.no_grad():
        bs = 512
        for i in range(0, N, bs):
            xb = torch.from_numpy(X[i:i+bs]).float().to(device)  # (B,T,C)
            out = model(xb)                                      # (logits, right, left)
            logits, pr, pl = out[:3]
            preds_r.append(pr.detach().cpu().numpy().reshape(-1))
            preds_l.append(pl.detach().cpu().numpy().reshape(-1))
            preds_cls.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    pred_r = np.concatenate(preds_r); pred_l = np.concatenate(preds_l)
    pred_cls = np.concatenate(preds_cls).astype(np.int64)

    # 分类准确率
    acc = (pred_cls == y_cls).mean()
    print(f"[Result] Activity Acc: {acc*100:.2f}% (N={N})")

    # 4) 绘图：角度两张叠加活动背景 + 底部活动序列
    x = np.arange(N)
    fig = plt.figure(figsize=(12, 8), dpi=500)

    # 右膝
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(x, y_r, label="GT Right", linewidth=2)
    ax1.plot(x, pred_r, label="Pred Right", linewidth=1.6)
    ax1.set_ylabel("Angle (°)")
    ax1.set_title("Right Knee Angle Tracking")
    ax1.grid(True, linestyle="--", alpha=0.3);
    ax1.legend()

    # 左膝
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(x, y_l, label="GT Left", linewidth=2)
    ax2.plot(x, pred_l, label="Pred Left", linewidth=1.6)
    ax2.set_ylabel("Angle (°)")
    ax2.set_title("Left Knee Angle Tracking")
    ax2.grid(True, linestyle="--", alpha=0.3);
    ax2.legend()

    # —— 在角度图上叠加活动背景（先 GT，再 Pred，Pred 更浅）
    if USE_ACTIVITY_BG:
        gt_segs = _segments_from_labels(y_cls)
        pred_segs = _segments_from_labels(pred_cls)

        def shade(ax, segs, layer_idx, label_prefix):
            import matplotlib.patches as mpatches
            legends = []
            for (s, e, cid) in segs:
                # x 方向是窗口索引，区间用 axvspan
                if not USE_GRAYSCALE:
                    ax.axvspan(s, e + 1, color=_color_for_class(cid, layer_idx), ymin=0, ymax=1)
                else:
                    # 灰度模式：不同类别固定几档灰度；Pred 用斜纹
                    gray_levels = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
                    g = gray_levels[cid % len(gray_levels)]
                    hatch = "//" if layer_idx == 1 else None
                    ax.axvspan(s, e + 1, color=str(g), alpha=0.25 if layer_idx == 0 else 0.15,
                               ymin=0, ymax=1, hatch=hatch, edgecolor="none")
            # 可选：做一个合成图例，避免太多 patch
            # 这里只放两条说明
            patch_gt = mpatches.Patch(facecolor=(0, 0, 0, 0.15), label=f"{label_prefix} Activity")
            return legends + [patch_gt]

        # GT 背景
        shade(ax1, gt_segs, layer_idx=0, label_prefix="GT")
        shade(ax2, gt_segs, layer_idx=0, label_prefix="GT")
        # Pred 背景（可关）
        if SHOW_PRED_BG:
            shade(ax1, pred_segs, layer_idx=1, label_prefix="Pred")
            shade(ax2, pred_segs, layer_idx=1, label_prefix="Pred")

    # 底部活动序列（数值阶梯）
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    acc = float((pred_cls == y_cls).mean())
    ax3.step(x, y_cls, where="mid", linewidth=1.5, label="GT Activity")
    ax3.step(x, pred_cls, where="mid", linewidth=1.2, linestyle="--",
             label=f"Pred Activity (Acc {acc * 100:.1f}%)")
    ax3.set_xlabel("Window Index",fontsize=12);
    ax3.set_ylabel("Class ID",fontsize=12)
    ax3.set_title("Activity Recognition",fontsize=12)
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend(loc="upper right",fontsize=12)

    # 可选：把 y 轴替换为类别名（仅当 ID 在 CLASS_NAMES 内）
    uniq = np.unique(np.concatenate([y_cls, pred_cls]))
    if all(int(u) in CLASS_NAMES for u in uniq):
        ax3.set_yticks(uniq)
        ax3.set_yticklabels([CLASS_NAMES[int(i)] for i in uniq], fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    print(f"图像已保存到: {FIG_PATH}")
    # plt.show()


if __name__ == "__main__":
    main()
