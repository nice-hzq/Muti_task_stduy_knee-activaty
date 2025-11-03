# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from model.model import  CNNLSTM_MultiTask
from train.train import train_model
# ==== 1. 加载所有受试者的数据 ====


def load_loso_subjects_merge_segments(data_root):
    subjects = {}

    for subject_folder in sorted(os.listdir(data_root)):
        subject_path = os.path.join(data_root, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        X_all, y_cls_all, y_reg_all = [], [], []

        for fname in sorted(os.listdir(subject_path)):
            if fname.endswith("_X.npy"):
                prefix = fname.replace("_X.npy", "")
                try:
                    x = np.load(os.path.join(subject_path, f"{prefix}_X.npy"))
                    y_cls = np.load(os.path.join(subject_path, f"{prefix}_y_cls.npy"))
                    y_reg = np.load(os.path.join(subject_path, f"{prefix}_y_reg.npy"))
                    X_all.append(x)
                    y_cls_all.append(y_cls)
                    y_reg_all.append(y_reg)
                except Exception as e:
                    print(f" Failed loading {prefix}: {e}")

        if X_all:
            X_all = np.concatenate(X_all, axis=0)
            y_cls_all = np.concatenate(y_cls_all, axis=0)
            y_reg_all = np.concatenate(y_reg_all, axis=0)
            subjects[subject_folder] = (X_all, y_cls_all, y_reg_all)
            print(f" Loaded subject {subject_folder}: X={X_all.shape}, y_cls={y_cls_all.shape}, y_reg={y_reg_all.shape}")
        else:
            print(f" No valid data for subject {subject_folder}")

    return subjects

def load_loso_subjects(data_root):
    subjects = {}
    for subject_folder in sorted(os.listdir(data_root)):
        subject_path = os.path.join(data_root, subject_folder)
        if not os.path.isdir(subject_path):
            continue
        try:
            X = np.load(os.path.join(subject_path, "X.npy"))
            y_cls = np.load(os.path.join(subject_path, "y_cls.npy"))
            y_reg = np.load(os.path.join(subject_path, "y_reg.npy"))
            subjects[subject_folder] = (X, y_cls, y_reg)
        except Exception as e:
            print(f" Error loading {subject_folder}: {e}")
    return subjects


# ==== 2. 构建 DataLoader ====
def make_dataloader(X, y_cls, y_reg, batch_size=64, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_cls_tensor = torch.tensor(y_cls, dtype=torch.long)
    y_reg_tensor = torch.tensor(y_reg, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_cls_tensor, y_reg_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ==== 3. LOSO 主循环 ====
def loso_evaluation(subjects_dict, model_class, model_kwargs=None,
                    batch_size=64, epochs=30, lr=1e-3, save_dir="logs/loso"):
    acc_list, rmse_r_list, rmse_l_list = [], [], []
    subject_ids = list(subjects_dict.keys())

    for i, test_id in enumerate(subject_ids):
        print(f"\n LOSO Fold {i+1}/{len(subject_ids)} - Test Subject: {test_id}")

        X_test, y_cls_test, y_reg_test = subjects_dict[test_id]
        X_train, y_cls_train, y_reg_train = [], [], []

        for sid, (X, y_cls, y_reg) in subjects_dict.items():
            if sid != test_id:
                X_train.append(X)
                y_cls_train.append(y_cls)
                y_reg_train.append(y_reg)

        X_train = np.concatenate(X_train, axis=0)
        y_cls_train = np.concatenate(y_cls_train, axis=0)
        y_reg_train = np.concatenate(y_reg_train, axis=0)

        # 验证集划分
        X_train, X_val, y_cls_train, y_cls_val, y_reg_train, y_reg_val = train_test_split(
            X_train, y_cls_train, y_reg_train, test_size=0.1, random_state=42
        )

        train_loader = make_dataloader(X_train, y_cls_train, y_reg_train, batch_size, shuffle=True)
        val_loader = make_dataloader(X_val, y_cls_val, y_reg_val, batch_size, shuffle=False)
        test_loader = make_dataloader(X_test, y_cls_test, y_reg_test, batch_size, shuffle=False)

        # 初始化模型
        model = model_class(input_size=X_train.shape[2], **(model_kwargs or {}))


        # 训练 & 测试（train_model 需支持 test_loader）
        _, _, _, acc, rmse_r, rmse_l = train_model(
            model, train_loader, val_loader, test_loader=test_loader,
            epochs=epochs, lr=lr, save_dir=os.path.join(save_dir, f"{test_id}")
        )

        acc_list.append(acc)
        rmse_r_list.append(rmse_r)
        rmse_l_list.append(rmse_l)

    # ==== 汇总 ====
    print("\n LOSO 测试汇总结果：")
    for i, sid in enumerate(subject_ids):
        print(f"{sid}: Acc={acc_list[i]:.4f}, RMSE_R={rmse_r_list[i]:.2f}, RMSE_L={rmse_l_list[i]:.2f}")

    print("\n 平均结果：")
    print(f"分类准确率: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"右膝角度 RMSE: {np.mean(rmse_r_list):.2f} ± {np.std(rmse_r_list):.2f}°")
    print(f"左膝角度 RMSE: {np.mean(rmse_l_list):.2f} ± {np.std(rmse_l_list):.2f}°")

if __name__ == "__main__":
    data_root = "/home/lenovo/PycharmProjects/CNN+LSTM/data/loso"
    subjects_data = load_loso_subjects_merge_segments(data_root)

    loso_evaluation(
        subjects_dict=subjects_data,
        model_class=CNNLSTM_MultiTask,
        model_kwargs={"cnn_channels": 64,  "num_classes": 7},
        batch_size=64,
        epochs=30,
        lr=1e-3,
        save_dir="logs/loso_results"
    )
