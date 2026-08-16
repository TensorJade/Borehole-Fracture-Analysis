# Reproducibility

## 当前训练记录

- 样本：300 对图像/掩码。
- 固定随机种子：42。
- 划分：240 对训练、60 对验证；验证集不做随机增强。
- 训练：100 epochs，Adam，初始学习率 `1e-4`，加权 BCE + Dice。
- 最佳 checkpoint：epoch 47，以验证 F1 选择。
- 最佳验证指标：pixel accuracy 0.9563、precision 0.3824、recall 0.5368、F1 0.4467。

训练历史保存在本地 `artifacts/segmentation/training_history.json`，权重保存在 `models/au_net_crack.pth`，两者默认不进入 Git。

## 可复现命令

```powershell
borehole-fracture check
borehole-fracture train --epochs 100 --batch-size 16
```

CUDA、PyTorch 版本、GPU 型号和底层算子可能导致细微数值差异。当前记录没有提供独立测试集或跨来源外部验证，因此这些指标只能描述一次内部固定划分，不代表泛化性能。

## 仍缺少的复现材料

- 论文原始训练/验证/测试样本 ID 清单。
- 数据集正式名称、版本、下载地址与许可。
- 论文表格对应的原始日志、checkpoint 校验值和运行环境锁定文件。
- 附件4人工复核掩码、裂隙平面真值与 JRC 现场测量值。
- 钻孔方位角、倾角和圆周零度方向；当前实现假设竖直孔。
