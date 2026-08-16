# Borehole Fracture Analysis

一个可安装、可测试的钻孔成像裂隙分析工程，覆盖 Attention U-Net 分割、裂隙正弦拟合、JRC 粗糙度估计、三维平面重构、连通性分析与补孔候选位置计算。

> 当前仓库适合发布“代码”。本地原始附件、300 组训练样本、模型权重和生成结果没有附带可验证的再分发许可，因此已由 `.gitignore` 排除，不能随代码直接上传 GitHub。详见 [数据说明](docs/data.md) 和 [NOTICE](NOTICE)。

## 功能模块

| 命令 | Python 模块 | 作用 |
|---|---|---|
| `segment` | `segmentation.py` | Attention U-Net 训练与裂隙掩码预测 |
| `fit-sinusoids` | `sinusoidal_fitting.py` | DBSCAN 聚类与正弦参数拟合 |
| `analyze-roughness` | `roughness_analysis.py` | 平整化、采样和 JRC 估计 |
| `reconstruct` | `reconstruction.py`、`connectivity.py` | 三维平面拟合、连通概率和补孔候选 |

## 安装

推荐 Python 3.10 或 3.11：

```powershell
git clone https://github.com/TensorJade/Borehole-Fracture-Analysis.git
cd borehole-fracture-analysis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

若只运行、不开发，可使用 `python -m pip install -e .`。

## 准备数据

按以下结构放置本地数据，文件名规则与完整字段见 [docs/data.md](docs/data.md)：

```text
data/
├── raw/
│   ├── segmentation-images/
│   ├── sinusoidal-fractures/
│   ├── roughness-images/
│   └── borehole-scans/hole-01/...
└── training/
    ├── images/<sample-id>.<ext>
    └── masks/<sample-id>_mask.png
```

模型默认路径为 `models/au_net_crack.pth`。

## 使用

```powershell
borehole-fracture check
borehole-fracture train --epochs 100 --batch-size 16
borehole-fracture segment
borehole-fracture fit-sinusoids
borehole-fracture analyze-roughness
borehole-fracture reconstruct
borehole-fracture run-all
```

未安装命令行入口时，也可以使用 `python -m borehole_fracture_analysis check`。Windows 用户还可运行 `run-project.bat check`。所有生成文件写入 `artifacts/`，不会覆盖原始数据。

## 开发与验证

```powershell
ruff check .
pytest --cov=borehole_fracture_analysis --cov-report=term-missing
```

架构、复现记录和开发约定分别见 [架构说明](docs/architecture.md)、[复现记录](docs/reproducibility.md)、[验证记录](docs/verification.md)、[依赖审计](docs/dependencies.md)、[发布清单](docs/release-checklist.md)、[开发规范](docs/development-standards.md) 和 [贡献指南](CONTRIBUTING.md)。

## 科学与工程边界

- 当前权重由本地 300 对样本按固定随机种子划分为 240 对训练、60 对验证得到；这不是独立外部验证。
- JRC 为图像经验估计，没有现场真值时不得解释为工程验收值。
- 连通概率和补孔坐标来自简化几何规则，只能作为候选方案，不能替代地质调查和现场决策。
- 精确复现论文表格仍需要原始固定划分、数据版本、标注说明、训练日志和现场真值。

## 许可

源代码采用 [MIT License](LICENSE)。数据、模型权重与输出不自动获得该许可。
