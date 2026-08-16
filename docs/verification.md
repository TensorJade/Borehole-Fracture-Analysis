# Verification Record

## 2026-08-16 local verification

- Environment: Windows, Python 3.12.6, editable install from `pyproject.toml`.
- Static analysis: `ruff check .` passed.
- Unit and synthetic pipeline tests: 24 passed.
- Coverage: 80.75% branch-aware project coverage; configured minimum is 80%.
- Runnable demo: CLI and Python example passed; the wheel contains both the demo module and JSON parameters.
- Local prerequisite check: four raw datasets found, 300 training pairs matched, checkpoint found.
- Real-data stages: 71 masks generated; 14 sinusoidal-fit rows; 130 roughness rows; 60 reconstructed planes; connectivity and three candidate locations generated.

GitHub Actions passed on Python 3.10 and 3.11 for the current `master` baseline. Repository branch-protection settings remain an external configuration item that must be checked in GitHub.
