# Multi-Task Knee Angle Prediction

基于肌电传感器信号的**多任务深度学习**项目，同时进行：
- **活动分类**：8 分类任务（MODE 0–7）
- **膝关节角度回归**：预测膝关节角度值

## 数据说明

| 列名 | 含义 | 用途 |
|------|------|------|
| LEFT_TA | 胫骨前肌 | 输入特征 |
| LEFT_MG | 腓肠肌内侧头 | 输入特征 |
| LEFT_SOL | 比目鱼肌 | 输入特征 |
| LEFT_BF | 股二头肌 | 输入特征 |
| LEFT_ST | 半腱肌 | 输入特征 |
| LEFT_VL | 股外侧肌 | 输入特征 |
| LEFT_RF | 股直肌 | 输入特征 |
| LEFT_KNEE | 膝关节角度 | 回归标签 |
| MODE | 活动模式 (0–7) | 分类标签 |

数据文件位于 `data/raw/`（示例）和 `../data/Processed/`（完整数据集）。

## 模型结构

```
Input: [batch, 7, 128]
    ↓
Shared CNN Encoder (Conv1D → BN → ReLU → Dropout × 2)
    ↓
BiLSTM Encoder
    ↓
Global Average Pooling
   ↙          ↘
Classification Head   Regression Head
[batch, 8]            [batch, 1]
```

## 环境安装

```bash
cd multitask_knee_angle
pip install -r requirements.txt
```

## 训练

### 单文件模式

```bash
python -m src.train \
    --csv_path data/raw/AB192_Circuit_033_post.csv \
    --window_size 128 \
    --stride 64 \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-3 \
    --lambda_reg 1.0
```

### 多文件模式（完整数据集）

```bash
python -m src.train \
    --csv_dir ../data/Processed \
    --batch_size 64 \
    --epochs 100
```

## 评估

```bash
python -m src.evaluate \
    --csv_path data/raw/AB192_Circuit_033_post.csv \
    --checkpoint outputs/checkpoints/best_model.pth
```

## 预测

```bash
python -m src.predict \
    --csv_path data/raw/new_data.csv \
    --checkpoint outputs/checkpoints/best_model.pth \
    --output outputs/predictions/pred.csv
```

## 测试

```bash
pytest tests/
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --csv_path | "" | 单个 CSV 文件路径 |
| --csv_dir | "" | 多个 CSV 文件目录 |
| --output_dir | outputs | 输出目录 |
| --window_size | 128 | 滑动窗口大小 |
| --stride | 64 | 窗口步长 |
| --batch_size | 64 | 批次大小 |
| --epochs | 100 | 训练轮数 |
| --lr | 1e-3 | 学习率 |
| --lambda_reg | 1.0 | 回归损失权重 |
| --hidden_size | 64 | LSTM 隐藏层大小 |
| --num_layers | 1 | LSTM 层数 |
| --dropout | 0.3 | Dropout 比例 |
| --scale_reg_target | True | 是否标准化回归目标 |
| --device | auto | 设备 (cuda/cpu) |

## 输出文件

训练完成后在 `outputs/` 下生成：

```
outputs/
├── logs/
│   ├── {timestamp}/
│   │   ├── train.log                  # 训练日志
│   │   ├── training_history.csv       # 每 epoch 指标
│   │   └── data_summary.json          # 数据概览
├── checkpoints/
│   └── best_model.pth                 # 最佳模型
├── metrics/
│   ├── test_metrics.json              # 分类+回归指标
│   ├── test_classification_report.csv # 逐类分类报告
│   └── test_regression_metrics.json   # 回归指标
├── figures/
│   ├── confusion_matrix.png           # 混淆矩阵
│   ├── knee_angle_curve.png           # 预测曲线
│   └── knee_angle_scatter.png         # 散点图
└── predictions/
    └── test_predictions.csv           # 测试集预测结果
```

## 常见问题

**Q: 如何修改窗口长度？**
使用 `--window_size` 和 `--stride` 参数。

**Q: 如何扩展到多个 CSV 文件？**
使用 `--csv_dir` 参数指定目录，所有 `.csv` 文件自动合并。

**Q: 数据泄漏如何防止？**
本项目采用时间顺序划分（70%/15%/15%），标准化器仅在训练集上拟合。

**Q: 如何更换模型结构？**
替换 `src/model.py` 中的 `CNNLSTMMultiTask` 类即可（如换成 Transformer、TCN）。
