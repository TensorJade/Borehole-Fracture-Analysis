# Dependency Review

运行时依赖统一维护在 `pyproject.toml`。下表记录 2026-08-16 本地审计用途，不替代发布时的许可证和安全扫描。

| 组件 | 用途 | 常见上游许可 | 处理结论 |
|---|---|---|---|
| NumPy / SciPy / pandas | 数值、优化和表格计算 | BSD 系列 | 可作为依赖，不复制其源码 |
| Matplotlib | 静态科学绘图 | Matplotlib License | 可作为依赖，发布时保留上游声明 |
| OpenCV | 图像读取、形态学和连通域 | Apache-2.0 | 可作为依赖，关注二进制分发声明 |
| Pillow | 图像缩放与保存 | HPND | 可作为依赖 |
| scikit-image / scikit-learn | 骨架、连通域、DBSCAN | BSD 系列 | 可作为依赖 |
| PyTorch / torchvision | Attention U-Net 训练与推理 | BSD 系列 | 可作为依赖；CUDA 二进制另有组件条款 |
| openpyxl | Excel 结果导出 | MIT | 可作为依赖 |

发布前需重新执行依赖清单、漏洞和许可证扫描，确认实际解析版本、传递依赖、CUDA 组件和平台二进制许可。当前版本采用下限约束而非完整锁文件，因此供应链可重复性仍属于待改进项。
