# Contributing

感谢参与改进。提交前请先阅读本文件和 `docs/development-standards.md`。

## 开发流程

1. 从最新发布分支拉取代码，创建个人功能分支，例如 `alice/feature_mask_export_20260816`。
2. 每次提交只完成一个可验证的目的，提交信息简明描述“做了什么、为什么”。
3. 提交 Pull Request 前运行 `ruff check .` 和 `pytest --cov=borehole_fracture_analysis`。
4. PR 必须说明需求、实现、验证证据、风险和回滚方式；至少一名非作者完成审查后再合并。
5. 禁止直接向受保护的 `master` 或发布分支推送。仓库维护者需在 GitHub 中实际配置保护规则。

## 代码约定

- Python 文件、包、函数使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`。
- 公共函数应有简洁 docstring；日志不得包含口令、密钥、完整用户隐私数据。
- 不得提交数据集、模型权重、生成结果、凭据或本机绝对路径。
- 新依赖需说明用途、许可证、维护状态与安全风险。
- 修复缺陷和新增算法都应补充相应测试。

## 问题与安全漏洞

一般缺陷请使用 Issue 模板。安全问题请按 `SECURITY.md` 私下报告，不要公开披露可利用细节。
