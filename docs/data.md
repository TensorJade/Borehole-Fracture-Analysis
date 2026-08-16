# Data and Naming

## 当前本地数据清单

| 语义目录 | 原附件 | 本地数量 | 公开发布状态 |
|---|---|---:|---|
| `data/raw/segmentation-images` | 附件1 | 10 | 不可发布，缺来源与许可 |
| `data/raw/sinusoidal-fractures` | 附件2 | 10 | 不可发布，缺来源与许可 |
| `data/raw/roughness-images` | 附件3 | 11 | 不可发布，缺来源与许可 |
| `data/raw/borehole-scans` | 附件4 | 40 | 不可发布，缺来源与许可 |
| `data/training/images` + `masks` | Q1_train | 300 对 | 不可发布，缺来源与许可 |
| `data/examples/images` | input/img | 27 | 不可发布，缺来源与许可 |

## 文件命名规则

- 分割图：`segmentation-01.jpg`
- 正弦裂隙图：`sinusoidal-01.jpg`
- 粗糙度图：`roughness-01.jpg`
- 钻孔目录：`hole-01`
- 深度图：`depth-00-01m.jpg`
- 训练掩码：必须与图像 stem 一致并增加 `_mask`，例如 `sample-001_mask.png`

## 贡献者补充数据时必须提供

1. 数据名称、来源 URL 或获取单位、版本/获取日期。
2. 原始许可文本或明确的书面再分发授权。
3. 标注方法、类别定义、质量控制和已知偏差。
4. 数据划分清单及避免同源泄漏的说明。
5. 若涉及敏感地点、人员或商业资料，提供脱敏与合规说明。

只有上述材料核验完成后，才可调整 `.gitignore` 或建立独立数据发布仓库。
