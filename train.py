from model.model import multi_task_loss
from datetime import datetime, time
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
from sklearn.metrics import accuracy_score, mean_squared_error


# 顶级日志输出目录
base_log_dir = "D:/exercise/CNN+LSTM/logs"


# 当前时间戳作为子目录名
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(base_log_dir, f"run_{time_str}")
os.makedirs(run_dir, exist_ok=True)

# 设置日志文件路径
log_path = os.path.join(run_dir, "train.log")

# 配置日志输出
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 如果你想保存图像时也用这个目录：
accuracy_fig_path = os.path.join(run_dir, "accuracy.png")
rmse_fig_path = os.path.join(run_dir, "rmse.png")

def plot_confusion_matrix(y_true, y_pred, save_path, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, colorbar=True, ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# def train_model(model, train_loader, val_loader, test_loader=None,
#                 epochs=20, lr=1e-3,
#                 device='cuda' if torch.cuda.is_available() else 'cpu',
#                 save_dir=None, model_name="best_model.pth",
#                 early_stop_patience=10):
#
#     best_metric = float("inf")
#     best_model_path = os.path.join(save_dir, model_name)
#     model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#     train_accs, val_accs = [], []
#
#
#     for epoch in range(1, epochs + 1):
#         model.train()
#         total_loss = 0
#         all_preds, all_labels = [], []
#
#         step = 0
#
#         for xb, yb_cls, yb_reg in train_loader:
#             step += 1  # 手动计数
#             xb = xb.to(device)
#             yb_cls = yb_cls.to(device)
#             yb_right = yb_reg[:, 0].to(device)
#             yb_left = yb_reg[:, 1].to(device)
#
#             out_cls, out_right, out_left = model(xb)
#             loss = multi_task_loss(out_cls, out_right, out_left, yb_cls, yb_right, yb_left)
#             # loss, w_cls, w_reg = multi_task_loss_discrete(
#             #     out_cls, out_right, out_left,
#             #     yb_cls, yb_right, yb_left,
#             #     mode="fixed_cls",  # "fixed_cls" / "fixed_reg" / "random"
#             #     weight_choices=[0.2, 0.3, 0.5]
#             # )
#             # uncert = UncertaintyParams().to(device)
#             # optimizer = torch.optim.Adam(
#             #     list(model.parameters()) + list(uncert.parameters()),
#             #     lr=1e-3
#             # )
#             # loss = multi_task_loss(out_cls, out_right, out_left,
#             #                        yb_cls, yb_right, yb_left,
#             #                        use_uncertainty=True,
#             #                        s_cls=uncert.s_cls, s_reg=uncert.s_reg)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # if step % 50 == 0:
#             #     print(f"[Epoch {epoch}] loss={loss.item():.4f}, w_cls={w_cls:.1f}, w_reg={w_reg:.1f}")
#
#             total_loss += loss.item()
#             all_preds.extend(out_cls.argmax(dim=1).cpu().numpy())
#             all_labels.extend(yb_cls.cpu().numpy())
#
#
#         train_acc = accuracy_score(all_labels, all_preds)
#         train_accs.append(train_acc)
#
#         # 验证阶段
#         model.eval()
#         val_preds, val_labels = [], []
#         right_true, right_pred = [], []
#         left_true, left_pred = [], []
#
#         with torch.no_grad():
#             for xb, yb_cls, yb_reg in val_loader:
#                 xb = xb.to(device)
#                 yb_cls = yb_cls.to(device)
#                 yb_right = yb_reg[:, 0].to(device)
#                 yb_left = yb_reg[:, 1].to(device)
#
#                 out_cls, out_right, out_left = model(xb)
#                 val_preds.extend(out_cls.argmax(dim=1).cpu().numpy())
#                 val_labels.extend(yb_cls.cpu().numpy())
#                 right_true.extend(yb_right.cpu().numpy())
#                 right_pred.extend(out_right.cpu().numpy())
#                 left_true.extend(yb_left.cpu().numpy())
#                 left_pred.extend(out_left.cpu().numpy())
#
#         val_acc = accuracy_score(val_labels, val_preds)
#         train_accs.append(train_acc)
#         val_accs.append(val_acc)
#         rmse_right = np.sqrt(mean_squared_error(right_true, right_pred))
#         rmse_left = np.sqrt(mean_squared_error(left_true, left_pred))
#
#         log_msg = (f"Epoch {epoch}/{epochs} | "
#                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
#                    f"RMSE(R): {rmse_right:.2f}, RMSE(L): {rmse_left:.2f})")
#         print(log_msg)
#         logging.info(log_msg)
#
#         current_metric = rmse_right + rmse_left - val_acc
#         # current_metric = rmse_right + rmse_left
#         if current_metric < best_metric:
#             best_metric = current_metric
#             epochs_no_improve = 0  #  重置无提升计数
#             torch.save(model.state_dict(), best_model_path)
#             logging.info(f"Best model saved at epoch {epoch} to {best_model_path}")
#         else:
#             epochs_no_improve += 1
#             logging.info(f"No improvement for {epochs_no_improve} epoch(s).")
#
#             # === 提前停止判断 ===
#         if epochs_no_improve >= early_stop_patience:
#             print(f" Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
#             logging.info(f" Early stopping at epoch {epoch}")
#             break
#
#     # === 保存分类准确率图 ===
#     acc_path = os.path.join(save_dir, "classification_accuracy.png")
#     plt.figure(figsize=(8, 5))
#     plt.plot(train_accs, label='Train Accuracy')
#     plt.plot(val_accs, label='Validation Accuracy')
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.title("Classification Accuracy over Epochs")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(acc_path)
#     plt.close()
#     logging.info(f"保存分类准确率图：{acc_path}")
#
#     # === 混淆矩阵图像 ===
#     class_names = [f"Act{i}" for i in range(7)]
#     confmat_path = os.path.join(save_dir, "confusion_matrix.png")
#     plot_confusion_matrix(val_labels, val_preds, confmat_path, class_names=class_names)
#     logging.info(f"保存混淆矩阵图像：{confmat_path}")
#
#     # === 加载最优模型并在测试集上评估（如果提供了 test_loader） ===
#     test_acc, test_rmse_r, test_rmse_l = None, None, None
#     if test_loader is not None:
#         state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
#         model.load_state_dict(state_dict)
#         model.to(device)
#         model.eval()
#
#         all_preds_cls, all_true_cls = [], []
#         right_preds, right_trues = [], []
#         left_preds, left_trues = [], []
#
#         with torch.no_grad():
#             for xb, yb_cls, yb_reg in test_loader:
#                 xb = xb.to(device)
#                 yb_cls = yb_cls.to(device)
#                 yb_right = yb_reg[:, 0].to(device)
#                 yb_left = yb_reg[:, 1].to(device)
#
#                 out_cls, out_right, out_left = model(xb)
#                 all_preds_cls.extend(out_cls.argmax(dim=1).cpu().numpy())
#                 all_true_cls.extend(yb_cls.cpu().numpy())
#                 right_preds.extend(out_right.cpu().numpy())
#                 right_trues.extend(yb_right.cpu().numpy())
#                 left_preds.extend(out_left.cpu().numpy())
#                 left_trues.extend(yb_left.cpu().numpy())
#
#         test_acc = accuracy_score(all_true_cls, all_preds_cls)
#         test_rmse_r = np.sqrt(mean_squared_error(right_trues, right_preds))
#         test_rmse_l = np.sqrt(mean_squared_error(left_trues, left_preds))
#
#         print(f"\n 测试集评估: Acc={test_acc:.4f}, RMSE_R={test_rmse_r:.2f}, RMSE_L={test_rmse_l:.2f}")
#         logging.info(f" 测试集评估结果 - 分类准确率: {test_acc:.4f}")
#         logging.info(f" 测试集右膝 RMSE: {test_rmse_r:.2f}°")
#         logging.info(f" 测试集左膝 RMSE: {test_rmse_l:.2f}°")
#
#     # === 返回评估指标 ===
#     return val_acc, rmse_right, rmse_left, test_acc, test_rmse_r, test_rmse_l

def train_model(model, train_loader, val_loader, test_loader=None,
                epochs=20, lr=1e-3,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                save_dir=None, model_name="best_model.pth",
                early_stop_patience=10):

    import time
    best_metric = float("inf")
    best_model_path = os.path.join(save_dir, model_name)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs, val_accs = [], []
    epochs_no_improve = 0  # ← NEW: 初始化

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        step = 0
        for xb, yb_cls, yb_reg in train_loader:
            step += 1
            xb = xb.to(device)
            yb_cls = yb_cls.to(device)
            yb_right = yb_reg[:, 0].to(device)
            yb_left  = yb_reg[:, 1].to(device)

            out_cls, out_right, out_left = model(xb)
            loss = multi_task_loss(out_cls, out_right, out_left, yb_cls, yb_right, yb_left)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            all_preds.extend(out_cls.argmax(dim=1).cpu().numpy())
            all_labels.extend(yb_cls.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        train_accs.append(train_acc)

        # ---------- 验证 ----------
        model.eval()
        val_preds, val_labels = [], []
        right_true, right_pred = [], []
        left_true,  left_pred  = [], []

        with torch.no_grad():
            for xb, yb_cls, yb_reg in val_loader:
                xb = xb.to(device)
                yb_cls = yb_cls.to(device)
                yb_right = yb_reg[:, 0].to(device)
                yb_left  = yb_reg[:, 1].to(device)

                out_cls, out_right, out_left = model(xb)
                val_preds.extend(out_cls.argmax(dim=1).cpu().numpy())
                val_labels.extend(yb_cls.cpu().numpy())
                right_true.extend(yb_right.cpu().numpy())
                right_pred.extend(out_right.cpu().numpy())
                left_true.extend(yb_left.cpu().numpy())
                left_pred.extend(out_left.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_accs.append(val_acc)
        rmse_right = np.sqrt(mean_squared_error(right_true, right_pred))
        rmse_left  = np.sqrt(mean_squared_error(left_true,  left_pred))

        log_msg = (f"Epoch {epoch}/{epochs} | "
                   f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
                   f"RMSE(R): {rmse_right:.2f}, RMSE(L): {rmse_left:.2f}")
        print(log_msg)
        logging.info(log_msg)

        # 早停指标
        current_metric = rmse_right + rmse_left - val_acc
        if current_metric < best_metric:
            best_metric = current_metric
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Best model saved at epoch {epoch} to {best_model_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= early_stop_patience:
            print(f" Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            logging.info(f" Early stopping at epoch {epoch}")
            break

    # ---------- 保存 acc 曲线 ----------
    acc_path = os.path.join(save_dir, "classification_accuracy.png")
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs,   label='Validation Accuracy')
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Classification Accuracy over Epochs")
    plt.legend(); plt.grid(True)
    plt.savefig(acc_path); plt.close()
    logging.info(f"保存分类准确率图：{acc_path}")

    # ---------- 保存验证集混淆矩阵 ----------
    class_names = [f"Act{i}" for i in range(7)]
    confmat_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_confusion_matrix(val_labels, val_preds, confmat_path, class_names=class_names)
    logging.info(f"保存混淆矩阵图像：{confmat_path}")

    # ---------- 测试集评估（新增 Corr / RMSE-Δ / InferenceTime） ----------
    test_acc = test_rmse_r = test_rmse_l = None
    if test_loader is not None:
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        all_preds_cls, all_true_cls = [], []
        right_preds, right_trues = [], []
        left_preds,  left_trues  = [], []
        infer_times = []  # 毫秒/样本

        with torch.no_grad():
            for xb, yb_cls, yb_reg in test_loader:
                xb = xb.to(device)
                yb_cls = yb_cls.to(device)
                yb_right = yb_reg[:, 0].to(device)
                yb_left  = yb_reg[:, 1].to(device)

                t0 = time.time()
                out_cls, out_right, out_left = model(xb)
                dt_ms_per_sample = (time.time() - t0) * 1000.0 / xb.size(0)  # ← NEW: 单样本推理时间
                infer_times.append(dt_ms_per_sample)

                all_preds_cls.extend(out_cls.argmax(dim=1).cpu().numpy())
                all_true_cls.extend(yb_cls.cpu().numpy())
                right_preds.extend(out_right.cpu().numpy())
                right_trues.extend(yb_right.cpu().numpy())
                left_preds.extend(out_left.cpu().numpy())
                left_trues.extend(yb_left.cpu().numpy())

        # numpy
        right_preds = np.asarray(right_preds).flatten()
        right_trues = np.asarray(right_trues).flatten()
        left_preds  = np.asarray(left_preds ).flatten()
        left_trues  = np.asarray(left_trues ).flatten()

        # 基础指标
        test_acc   = accuracy_score(all_true_cls, all_preds_cls)
        test_rmse_r = np.sqrt(mean_squared_error(right_trues, right_preds))
        test_rmse_l = np.sqrt(mean_squared_error(left_trues,  left_preds))

        # === NEW: 相关性 ===
        test_corr_r = float(np.corrcoef(right_trues, right_preds)[0, 1])
        test_corr_l = float(np.corrcoef(left_trues,  left_preds )[0, 1])

        # === NEW: 时间一致性（RMSE-Δ） ===
        diff_true_r, diff_pred_r = np.diff(right_trues), np.diff(right_preds)
        diff_true_l, diff_pred_l = np.diff(left_trues ), np.diff(left_preds )
        test_rmsed_r = float(np.sqrt(np.mean((diff_true_r - diff_pred_r) ** 2)))
        test_rmsed_l = float(np.sqrt(np.mean((diff_true_l - diff_pred_l) ** 2)))

        # === NEW: 平均单样本推理时间 ===
        avg_infer_ms = float(np.mean(infer_times))

        # 打印 + 日志
        print("\n===== 测试集评估 =====")
        print(f"Acc={test_acc:.4f} | RMSE_R={test_rmse_r:.3f}°, RMSE_L={test_rmse_l:.3f}°")
        print(f"Corr_R={test_corr_r:.3f}, Corr_L={test_corr_l:.3f} | RMSEΔ_R={test_rmsed_r:.3f}, RMSEΔ_L={test_rmsed_l:.3f}")
        print(f"Avg Inference Time: {avg_infer_ms:.3f} ms/sample")
        print("================================\n")

        logging.info(f" 测试集评估 - Acc: {test_acc:.4f}")
        logging.info(f" 测试集右膝 RMSE: {test_rmse_r:.3f}°, 左膝 RMSE: {test_rmse_l:.3f}°")
        logging.info(f" 测试集相关性 右: {test_corr_r:.3f}, 左: {test_corr_l:.3f}")
        logging.info(f" 测试集时间一致性 RMSEΔ 右: {test_rmsed_r:.3f}, 左: {test_rmsed_l:.3f}")
        logging.info(f" 平均推理时间: {avg_infer_ms:.3f} ms/样本")

    # 返回值保持不变（兼容你现有调用）
    return val_acc, rmse_right, rmse_left, test_acc, test_rmse_r, test_rmse_l

