import logging
import os
from datetime import datetime, time
from model.model import CNNLSTM_MultiTask, CNNLSTM_MultiTask, CNNLSTM_MultiTask_CNNOnly, CNNLSTM_MultiTask_LSTMOnly
from model.transform import TransformerMultiTask
from train.train import train_model
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import time  # 用于 time.time()
from datetime import datetime  # 用于日志目录命名
from model.CNNLSTM import CNNLSTM_MultiTask


# ==== 导入模型和训练函数 ====
# 确保你已经运行了 CNNLSTM_MultiTask 和 train_model 的定义

# ==== 加载数据 ====

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_data(npy_dir, max_files=150):
    X_list, y_cls_list, y_reg_list = [], [], []
    file_count = 0

    for fname in os.listdir(npy_dir):
        if not fname.endswith("_X.npy"):
            continue
        if file_count >= max_files:
            break

        base = fname.replace("_X.npy", "")
        try:
            x = np.load(os.path.join(npy_dir, f"{base}_X.npy")).astype(np.float32)
            y_cls = np.load(os.path.join(npy_dir, f"{base}_y_cls.npy")).astype(np.int64)
            y_reg = np.load(os.path.join(npy_dir, f"{base}_y_reg.npy")).astype(np.float32)
        except Exception as e:
            print(f" 跳过文件 {base}: {e}")
            continue

        X_list.append(x)
        y_cls_list.append(y_cls)
        y_reg_list.append(y_reg)
        file_count += 1
        print(f" 加载 {base}: {x.shape[0]} samples")

    X = np.concatenate(X_list, axis=0)
    y_cls = np.concatenate(y_cls_list, axis=0)
    y_reg = np.concatenate(y_reg_list, axis=0)
    return X, y_cls, y_reg


# def get_dataloaders_kfold(X, y_cls, y_reg, batch_size=128, fold_idx=1, n_splits=10, seed=42):
#     """
#     十折交叉验证划分：从 n_splits 中选第 fold_idx 折作为测试集，剩下的再划分出一折作为验证集。
#     返回：train_loader, val_loader, test_loader
#     """
#     # 保证 numpy 格式
#     X = np.array(X)
#     y_cls = np.array(y_cls)
#     y_reg = np.array(y_reg)
#
#     # 第一步：用 StratifiedKFold 拿到所有 10 折
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     all_indices = list(skf.split(X, y_cls))
#
#     if fold_idx >= len(all_indices):
#         raise ValueError(f"fold_idx 超出范围：应在 [0, {n_splits - 1}]，你给的是 {fold_idx}")
#
#     # 第二步：取当前 fold 作为测试集
#     test_idx = all_indices[fold_idx][1]
#     remain_idx = all_indices[fold_idx][0]  # 剩下的 9 折
#
#     X_test = X[test_idx]
#     y_cls_test = y_cls[test_idx]
#     y_reg_test = y_reg[test_idx]
#
#     # 第三步：对剩下的 9 折数据再做一次 9 折交叉验证，取其中一折为 val，其余为 train
#     skf_remain = StratifiedKFold(n_splits=n_splits - 1, shuffle=True, random_state=seed + 1)
#     val_subidx, train_subidx = next(iter(skf_remain.split(X[remain_idx], y_cls[remain_idx])))
#
#     val_idx = remain_idx[val_subidx]
#     train_idx = remain_idx[train_subidx]
#
#     def make_loader(X_part, y_cls_part, y_reg_part, shuffle=False):
#         dataset = TensorDataset(
#             torch.tensor(X_part, dtype=torch.float32),
#             torch.tensor(y_cls_part, dtype=torch.long),
#             torch.tensor(y_reg_part, dtype=torch.float32)
#         )
#         return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#
#     train_loader = make_loader(X[train_idx], y_cls[train_idx], y_reg[train_idx], shuffle=True)
#     val_loader = make_loader(X[val_idx], y_cls[val_idx], y_reg[val_idx], shuffle=False)
#     test_loader = make_loader(X_test, y_cls_test, y_reg_test, shuffle=False)
#
#     return train_loader, val_loader, test_loader

import inspect
import torch.nn as nn

import os, csv, math, json, random, statistics as stats
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import inspect

# —— 兼容“类/实例”的构建器（防止把 input_size 传进 forward）
def build_model(model_class_or_instance, input_size, **model_kwargs):
    if isinstance(model_class_or_instance, nn.Module):
        return model_class_or_instance
    sig = inspect.signature(model_class_or_instance.__init__)
    if 'input_size' in sig.parameters and 'input_size' not in model_kwargs:
        model_kwargs = {**model_kwargs, 'input_size': input_size}
    return model_class_or_instance(**model_kwargs)

def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mean_std_str(values):
    if not values: return "–"
    m = stats.mean(values); s = stats.pstdev(values) if len(values)>1 else 0.0
    return f"{m:.4f} ± {s:.4f}"

def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_model(model_class_or_instance, input_size, **model_kwargs):
    """
    兼容传进来的是：
      1) 类（如 TransformerMultiTask）
      2) 实例（如 TransformerMultiTask(...) 已经 new 好了）
    并且只在构造函数支持时才传 input_size，避免误传。
    """
    # 如果已经是实例，直接返回（不再调用）
    if isinstance(model_class_or_instance, nn.Module):
        return model_class_or_instance

    # 如果是类，检查 __init__ 是否有 input_size 这个参数
    sig = inspect.signature(model_class_or_instance.__init__)
    if 'input_size' in sig.parameters and 'input_size' not in model_kwargs:
        model_kwargs = {**model_kwargs, 'input_size': input_size}

    return model_class_or_instance(**model_kwargs)

def kfold_train_and_eval(X, y_cls, y_reg, model_class, model_kwargs=None,
                         n_splits=10, batch_size=128, epochs=20, lr=1e-3, base_logdir="logs"):
    # 统一创建一次主日志文件夹（带时间戳）
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_root_dir = os.path.join(base_logdir, f"run_{time_stamp}")
    os.makedirs(run_root_dir, exist_ok=True)

    if model_kwargs is None:
        model_kwargs = {}

    acc_list, rmse_r_list, rmse_l_list = [], [], []

    for fold in range(n_splits):
        print(f"\n Fold {fold + 1}/{n_splits}")

        # 每一折的子文件夹
        run_dir = os.path.join(run_root_dir, f"fold_{fold + 1}")
        os.makedirs(run_dir, exist_ok=True)
        # === 划分数据 ===
        train_loader, val_loader, test_loader = get_dataloaders_kfold(
            X, y_cls, y_reg,
            batch_size=batch_size,
            fold_idx=fold,
            n_splits=n_splits
        )

        # === 初始化模型 ===
        model = build_model(model_class, input_size=X.shape[2], **model_kwargs)


        # === 训练 ===
        val_acc, val_rmse_r, val_rmse_l, acc, rmse_r, rmse_l = train_model(
            model, train_loader, val_loader, test_loader,
            epochs=epochs, lr=lr, save_dir=run_dir, model_name="best_model.pth", early_stop_patience=30
        )

        # === 收集结果 ===
        acc_list.append(acc)
        rmse_r_list.append(rmse_r)
        rmse_l_list.append(rmse_l)

    # === 汇总输出 ===
    print("\n 十折交叉验证结果汇总：")
    for i in range(n_splits):
        print(f"Fold {i + 1}: Acc={acc_list[i]:.4f}, RMSE_R={rmse_r_list[i]:.2f}, RMSE_L={rmse_l_list[i]:.2f}")

    print("\n 平均结果：")
    print(f"分类准确率: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"右膝角度 RMSE: {np.mean(rmse_r_list):.2f} ± {np.std(rmse_r_list):.2f}°")
    print(f"左膝角度 RMSE: {np.mean(rmse_l_list):.2f} ± {np.std(rmse_l_list):.2f}°")

    # 日志保存
    result_log = os.path.join(run_root_dir, "summary.txt")
    with open(result_log, "w") as f:
        f.write(" 十折交叉验证结果汇总：\n")
        for i in range(n_splits):
            f.write(f"Fold {i + 1}: Acc={acc_list[i]:.4f}, RMSE_R={rmse_r_list[i]:.2f}, RMSE_L={rmse_l_list[i]:.2f}\n")
        f.write("\n 平均结果：\n")
        f.write(f"分类准确率: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}\n")
        f.write(f"右膝角度 RMSE: {np.mean(rmse_r_list):.2f} ± {np.std(rmse_r_list):.2f}°\n")
        f.write(f"左膝角度 RMSE: {np.mean(rmse_l_list):.2f} ± {np.std(rmse_l_list):.2f}°\n")

    return acc_list, rmse_r_list, rmse_l_list
def get_dataloaders_kfold(
    X, y_cls, y_reg,
    batch_size=128,
    fold_idx=0,          # 外层：选第几折做test
    n_splits=10,
    seed=42,
    val_fold_idx=0       # 内层：在剩余9折里选第几折做val
):
    """
    十折交叉验证：
      - 外层：从 n_splits 中选第 fold_idx 折作为测试集
      - 内层：对剩余 9 折再做 9 折切分，选第 val_fold_idx 折作为验证集，其余为训练集
    返回：train_loader, val_loader, test_loader
    """
    # 转为numpy
    X = np.asarray(X)
    y_cls = np.asarray(y_cls)
    y_reg = np.asarray(y_reg)

    # 保障y_cls是一维标签
    if y_cls.ndim > 1:
        y_cls = y_cls.argmax(axis=-1)

    skf_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_indices = list(skf_outer.split(X, y_cls))

    if not (0 <= fold_idx < n_splits):
        raise ValueError(f"fold_idx 超出范围：[0, {n_splits-1}]，得到 {fold_idx}")

    # 外层：当前折的test索引；其余为remain
    train_outer_idx, test_idx = all_indices[fold_idx]
    remain_idx = train_outer_idx  # 9折

    X_test = X[test_idx]
    y_cls_test = y_cls[test_idx]
    y_reg_test = y_reg[test_idx]

    # 内层：对remain再做9折，选其中一折做val
    skf_inner = StratifiedKFold(n_splits=n_splits - 1, shuffle=True, random_state=seed + 1)
    inner_splits = list(skf_inner.split(X[remain_idx], y_cls[remain_idx]))

    if not (0 <= val_fold_idx < (n_splits - 1)):
        raise ValueError(f"val_fold_idx 超出范围：[0, {n_splits-2}]，得到 {val_fold_idx}")

    inner_train_sub, inner_val_sub = inner_splits[val_fold_idx]  # 注意顺序：train, test
    # 映射回全局索引
    train_idx = remain_idx[inner_train_sub]  # 8折
    val_idx   = remain_idx[inner_val_sub]    # 1折

    def make_loader(X_part, y_cls_part, y_reg_part, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X_part, dtype=torch.float32),
            torch.tensor(y_cls_part, dtype=torch.long),
            torch.tensor(y_reg_part, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X[train_idx], y_cls[train_idx], y_reg[train_idx], shuffle=True)
    val_loader   = make_loader(X[val_idx],   y_cls[val_idx],   y_reg[val_idx],   shuffle=False)
    test_loader  = make_loader(X_test,       y_cls_test,       y_reg_test,       shuffle=False)

    return train_loader, val_loader, test_loader

def measure_inference_time(model, sample_input, device='cpu', repeat=100):
    model.eval()
    sample_input = sample_input.to(device)
    times = []

    with torch.no_grad():
        for _ in range(repeat):
            start = time.time()
            _ = model(sample_input)
            end = time.time()
            times.append((end - start) * 1000)  # ms

    return sum(times) / len(times)

# ====== 消融/对比实验清单 ======
# 每个条目：name（用于保存与打印）、model_class、model_kwargs
EXPERIMENTS = [
    # --- 你的 CNN+LSTM 基线 ---
    # dict(
    #     name="Base(CNNLSTM:Attn+SE+dilated+residual)",
    #     model_class=CNNLSTM_MultiTask,
    #     model_kwargs=dict(
    #         num_classes=7,
    #         cnn_channels=64, lstm_hidden=128,
    #         cnn_depth=3, cnn_dropout=0.1,
    #         lstm_layers=2, lstm_dropout=0.2, se_reduction=8,
    #         use_se=True, use_attn=True, attn_alt="last",
    #         use_dilation=True, use_residual=True
    #     )
    # ),
    dict(
        name="(CNN)",
        model_class=CNNLSTM_MultiTask_LSTMOnly,
        model_kwargs=dict(
            num_classes=7,
            cnn_channels=64, lstm_hidden=128,
            cnn_depth=3, cnn_dropout=0.1,
            lstm_layers=2, lstm_dropout=0.2, se_reduction=8,
            use_se=True, use_attn=True, attn_alt="last",
            use_dilation=True, use_residual=True
        )
    ),
    # # --- w/o Attn ---
    # dict(
    #     name="w/o Attn(lastpool)",
    #     model_class=CNNLSTM_MultiTask,
    #     model_kwargs=dict(
    #         num_classes=7,
    #         cnn_channels=64, lstm_hidden=128,
    #         cnn_depth=3, cnn_dropout=0.1,
    #         lstm_layers=2, lstm_dropout=0.2, se_reduction=8,
    #         use_se=True, use_attn=False, attn_alt="last",
    #         use_dilation=True, use_residual=True
    #     )
    # ),
    # # --- w/o SE ---
    # dict(
    #     name="w/o SE",
    #     model_class=CNNLSTM_MultiTask,
    #     model_kwargs=dict(
    #         num_classes=7,
    #         cnn_channels=64, lstm_hidden=128,
    #         cnn_depth=3, cnn_dropout=0.1,
    #         lstm_layers=2, lstm_dropout=0.2, se_reduction=8,
    #         use_se=False, use_attn=True, attn_alt="last",
    #         use_dilation=True, use_residual=True
    #     )
    # ),
    # # --- No dilation ---
    # dict(
    #     name="No-dilation",
    #     model_class=CNNLSTM_MultiTask,
    #     model_kwargs=dict(
    #         num_classes=7,
    #         cnn_channels=64, lstm_hidden=128,
    #         cnn_depth=3, cnn_dropout=0.1,
    #         lstm_layers=2, lstm_dropout=0.2, se_reduction=8,
    #         use_se=True, use_attn=True, attn_alt="last",
    #         use_dilation=False, use_residual=True
    #     )
    # ),
    # # --- No residual ---
    # dict(
    #     name="No-residual",
    #     model_class=CNNLSTM_MultiTask,
    #     model_kwargs=dict(
    #         num_classes=7,
    #         cnn_channels=64, lstm_hidden=128,
    #         cnn_depth=3, cnn_dropout=0.1,
    #         lstm_layers=2, lstm_dropout=0.2, se_reduction=8,
    #         use_se=True, use_attn=True, attn_alt="last",
    #         use_dilation=True, use_residual=False
    #     )
    # ),
    # --- Transformer baseline ---
    # dict(
    #     name="Transformer(attn-pool)",
    #     model_class=TransformerMultiTask,
    #     model_kwargs=dict(
    #         num_classes=7,
    #         d_model=128, nhead=4, num_layers=3,
    #         dim_feedforward=256, dropout=0.1,
    #         pooling='attn'
    #     )
    # ),
    # 还可加 'Transformer(mean)', 'Transformer(CLS)' 等
]
def run_all_ablation(X, y_cls, y_r, y_l,
                     seeds=(0,1,2),
                     out_dir="results_ablation",
                     kfold_args=None,
                     train_args=None):
    """
    X: (N,T,F), y_cls: (N,), y_r/y_l: (N,)
    seeds: 多次重复取均值±方差
    kfold_args/train_args: 透传到你的 kfold_train_and_eval（如折数、epoch、lr、batch 等）
    """
    os.makedirs(out_dir, exist_ok=True)
    tag = now_tag()

    csv_path = os.path.join(out_dir, f"ablation_{tag}.csv")
    md_path  = os.path.join(out_dir, f"ablation_{tag}.md")

    headers = ["ExpName", "Params(M)", "SD_Acc_mean±std", "SD_RMSE_R_mean±std", "SD_RMSE_L_mean±std"]

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv); writer.writerow(headers)

        md_lines = []
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("|" + "|".join(["---"]*len(headers)) + "|")

        for exp in EXPERIMENTS:
            name = exp["name"]
            model_class = exp["model_class"]
            model_kwargs = dict(exp["model_kwargs"])  # copy

            print(f"\n========== Running: {name} ==========")
            sd_acc_all, sd_rmse_r_all, sd_rmse_l_all = [], [], []
            params_m = None

            for seed in seeds:
                print(f"  Seed {seed}")
                set_global_seed(seed)

                # 构建模型（自动注入 input_size）
                model = build_model(model_class, input_size=X.shape[2], **model_kwargs)

                # 统计参数量（M）
                if params_m is None:
                    p = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    params_m = p / 1e6

                # 跑 K 折
                acc_list, rmse_r_list, rmse_l_list = kfold_train_and_eval(
                    model=model,
                    X=X, y_cls=y_cls, y_r=y_r, y_l=y_l,
                    **(kfold_args or {}), **(train_args or {})
                )

                # 汇总（如果你的 kfold 返回每折指标，这里对折再求均值）
                sd_acc_all.append(float(np.mean(acc_list)))
                sd_rmse_r_all.append(float(np.mean(rmse_r_list)))
                sd_rmse_l_all.append(float(np.mean(rmse_l_list)))

            row = [
                name,
                f"{params_m:.2f}",
                mean_std_str(sd_acc_all),
                mean_std_str(sd_rmse_r_all),
                mean_std_str(sd_rmse_l_all),
            ]
            writer.writerow(row)
            md_lines.append("| " + " | ".join(row) + " |")

    with open(md_path, "w", encoding="utf-8") as fmd:
        fmd.write("\n".join(md_lines))

    print(f"\n已保存：\n- CSV: {csv_path}\n- Markdown: {md_path}")
# EXPERIMENTS.append(
#     dict(
#         name="Base+UncertaintyWeighting",
#         model_class=CNNLSTM_MultiTask,
#         model_kwargs=dict(
#             num_classes=7,
#             cnn_channels=64, lstm_hidden=128,
#             cnn_depth=3, cnn_dropout=0.1,
#             lstm_layers=2, lstm_dropout=0.2, se_reduction=8,
#             use_se=True, use_attn=True, attn_alt="last",
#             use_dilation=True, use_residual=True
#         ),
#         # 额外的训练参数也可以塞到这里，然后在 run_all_ablation 里合并
#         # e.g., "train_overrides": dict(loss_mode="uncertainty")
#     )
# )

# def main():
#     # 1. 加载数据
#     npy_dir = "/home/lenovo/PycharmProjects/CNN+LSTM/data/AB156/NPY_MUTI"
#     log_base = "/home/lenovo/PycharmProjects/CNN+LSTM/data/AB156/logs"
#     X, y_cls, y_reg = load_data(npy_dir, max_files=150)
#     print(f"Loaded: X={X.shape}, y_cls={y_cls.shape}, y_reg={y_reg.shape}")
#
#     # 2. 参数统计
#     model = CNNLSTM_MultiTask(input_size=X.shape[2])
#     # model = TransformerMultiTask(
#     #     input_size=44,
#     #     num_classes=7,
#     #     d_model=128,  # ≈ 你的 LSTM 隐层
#     #     nhead=4,  # 128/4=32 维每头
#     #     num_layers=3,
#     #     dim_feedforward=256,  # 通常 2×~4× d_model
#     #     dropout=0.1,
#     #     pooling='attn'  # 与你原模型效果更接近；也方便做“w/o Attn”对比
#     # )
#
#     total_params, trainable_params = count_parameters(model)
#     print(f"模型总参数量: {total_params:,}")
#     print(f"可训练参数量: {trainable_params:,}")
#     logging.info(f"模型总参数量: {total_params:,}")
#     logging.info(f"可训练参数量: {trainable_params:,}")
#
#     # 3. 推理时间测量
#     example_input = torch.tensor(X[:1], dtype=torch.float32)  # 单条样本
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.to(device)
#     example_input = example_input.to(device)
#     avg_time_ms = measure_inference_time(model, example_input, device=device)
#     print(f"平均推理时间: {avg_time_ms:.2f} ms")
#     logging.info(f"平均推理时间: {avg_time_ms:.2f} ms")
#
#     # 4. 十折交叉验证
#     acc_list, rmse_r_list, rmse_l_list = kfold_train_and_eval(
#         X, y_cls, y_reg,
#         model_class=model,
#         n_splits=10,
#         batch_size=128,
#         epochs=300,
#         lr=1e-3,
#         base_logdir=log_base
#     )

def main():
    # 1. 加载数据 ----------------------------------------------------------
    # npy_dir = "/home/lenovo/PycharmProjects/CNN+LSTM/data/AB156/NPY_MUTI"
    # log_base = "/home/lenovo/PycharmProjects/CNN+LSTM/data/AB156/logs"
    # 需要循环的被试编号
    # subjects = ["AB156", "AB185", "AB186", "AB188", "AB189",
    #             "AB190", "AB191", "AB192", "AB193", "AB194"]
    subjects = ["AB156"]

    base_root = "/home/lenovo/PycharmProjects/CNN+LSTM/data"
    all_results = []  # 记录每个人的结果

    for sid in subjects:
        print(f"\n================= 当前受试者: {sid} =================")

        # 1. 生成路径
        npy_dir = os.path.join(base_root, sid, "NPY_MUTI")
        log_base = os.path.join(base_root, sid, "logs")
        os.makedirs(log_base, exist_ok=True)

        # 2. 加载该受试者的数据
        X, y_cls, y_reg = load_data(npy_dir, max_files=150)
        print(f"{sid}: X={X.shape}, y_cls={y_cls.shape}, y_reg={y_reg.shape}")

        # 2. 参数统计 + 推理时间（可只用一次基线模型） --------------------------
        base_model = CNNLSTM_MultiTask(input_size=X.shape[2])
        total_params, trainable_params = count_parameters(base_model)
        print(f"模型总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        logging.info(f"模型总参数量: {total_params:,}")
        logging.info(f"可训练参数量: {trainable_params:,}")

        # 推理时间测量
        example_input = torch.tensor(X[:1], dtype=torch.float32)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        base_model.to(device)
        example_input = example_input.to(device)
        avg_time_ms = measure_inference_time(base_model, example_input, device=device)
        print(f"平均推理时间: {avg_time_ms:.2f} ms")
        logging.info(f"平均推理时间: {avg_time_ms:.2f} ms")
        # 4. 逐个跑十折交叉验证 -------------------------------------------------
        results = []  # 收集每个实验的平均指标

        for exp in EXPERIMENTS:
            name = exp["name"]
            model_class = exp["model_class"]
            model_kwargs = exp["model_kwargs"]

            print(f"\n========== Running Experiment: {name} ==========")
            logging.info(f"========== Running Experiment: {name} ==========")

            model = model_class(**model_kwargs)
            # print(model)

            # 调用你现有的 K 折函数
            acc_list, rmse_r_list, rmse_l_list = kfold_train_and_eval(
                X, y_cls, y_reg,
                model_class=model,  # 这里直接传模型实例
                n_splits=10,
                batch_size=128,
                epochs=300,
                lr=1e-3,
                base_logdir=os.path.join(log_base, name.replace("/", "_"))
            )

            acc_mean = np.mean(acc_list)
            rmse_r_mean = np.mean(rmse_r_list)
            rmse_l_mean = np.mean(rmse_l_list)
            print(f"结果: Acc={acc_mean:.4f}, RMSE_R={rmse_r_mean:.4f}, RMSE_L={rmse_l_mean:.4f}")

            results.append(dict(
                ExpName=name,
                Acc=acc_mean,
                RMSE_R=rmse_r_mean,
                RMSE_L=rmse_l_mean
            ))
            # 5. 计算平均结果
            acc_mean = np.mean(acc_list)
            rmse_r_mean = np.mean(rmse_r_list)
            rmse_l_mean = np.mean(rmse_l_list)

            print(f" {sid} 结果: Acc={acc_mean:.4f}, RMSE_R={rmse_r_mean:.4f}, RMSE_L={rmse_l_mean:.4f}")

            # 保存结果
            all_results.append({
                "subject": sid,
                "Acc": acc_mean,
                "RMSE_R": rmse_r_mean,
                "RMSE_L": rmse_l_mean
            })

            # 6. 汇总所有受试者结果
        print("\n========== 所有受试者汇总结果 ==========")
        for res in all_results:
            print(f"{res['subject']} -> Acc={res['Acc']:.4f}, RMSE_R={res['RMSE_R']:.4f}, RMSE_L={res['RMSE_L']:.4f}")

        # 可选：保存到 CSV
        out_csv = os.path.join(base_root, "summary_all_subjects0.2_1.csv")
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["subject", "Acc", "RMSE_R", "RMSE_L"])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n 所有结果已保存到: {out_csv}")



    # X, y_cls, y_reg = load_data(npy_dir, max_files=150)
    # print(f"Loaded: X={X.shape}, y_cls={y_cls.shape}, y_reg={y_reg.shape}")

    # # 2. 参数统计 + 推理时间（可只用一次基线模型） --------------------------
    # base_model = CNNLSTM_MultiTask(input_size=X.shape[2])
    # total_params, trainable_params = count_parameters(base_model)
    # print(f"模型总参数量: {total_params:,}")
    # print(f"可训练参数量: {trainable_params:,}")
    # logging.info(f"模型总参数量: {total_params:,}")
    # logging.info(f"可训练参数量: {trainable_params:,}")
    #
    # # 推理时间测量
    # example_input = torch.tensor(X[:1], dtype=torch.float32)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # base_model.to(device)
    # example_input = example_input.to(device)
    # avg_time_ms = measure_inference_time(base_model, example_input, device=device)
    # print(f"平均推理时间: {avg_time_ms:.2f} ms")
    # logging.info(f"平均推理时间: {avg_time_ms:.2f} ms")
    # # 4. 逐个跑十折交叉验证 -------------------------------------------------
    # results = []  # 收集每个实验的平均指标
    #
    # for exp in EXPERIMENTS:
    #     name = exp["name"]
    #     model_class = exp["model_class"]
    #     model_kwargs = exp["model_kwargs"]
    #
    #     print(f"\n========== Running Experiment: {name} ==========")
    #     logging.info(f"========== Running Experiment: {name} ==========")
    #
    #     model = model_class(**model_kwargs)
    #     print(model)
    #
    #     # 调用你现有的 K 折函数
    #     acc_list, rmse_r_list, rmse_l_list = kfold_train_and_eval(
    #         X, y_cls, y_reg,
    #         model_class=model,  # 这里直接传模型实例
    #         n_splits=3,
    #         batch_size=128,
    #         epochs=3,
    #         lr=1e-3,
    #         base_logdir=os.path.join(log_base, name.replace("/", "_"))
    #     )
    #
    #     acc_mean = np.mean(acc_list)
    #     rmse_r_mean = np.mean(rmse_r_list)
    #     rmse_l_mean = np.mean(rmse_l_list)
    #     print(f"结果: Acc={acc_mean:.4f}, RMSE_R={rmse_r_mean:.4f}, RMSE_L={rmse_l_mean:.4f}")
    #
    #     results.append(dict(
    #         ExpName=name,
    #         Acc=acc_mean,
    #         RMSE_R=rmse_r_mean,
    #         RMSE_L=rmse_l_mean
    #     ))

    # 5. 保存汇总结果 --------------------------------------------------------
    # os.makedirs(log_base, exist_ok=True)
    # summary_csv = os.path.join(log_base, f"ablation_summary_uncertainty{now_tag()}.csv")
    # with open(summary_csv, "w", encoding="utf-8") as f:
    #     writer = csv.DictWriter(f, fieldnames=["ExpName", "Acc", "RMSE_R", "RMSE_L"])
    #     writer.writeheader()
    #     for row in results:
    #         writer.writerow(row)
    # print(f"\n所有实验完成，结果已保存：{summary_csv}")



if __name__ == "__main__":
    main()
