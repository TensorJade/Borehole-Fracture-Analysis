# Verification Record

## 2026-08-16 local verification

- Environment: Windows, Python 3.12.6, editable install from `pyproject.toml`.
- Static analysis: `ruff check .` passed.
- Unit and synthetic pipeline tests: 21 passed.
- Coverage: 80.15% branch-aware project coverage; configured minimum is 80%.
- Local prerequisite check: four raw datasets found, 300 training pairs matched, checkpoint found.
- Real-data stages: 71 masks generated; 14 sinusoidal-fit rows; 130 roughness rows; 60 reconstructed planes; connectivity and three candidate locations generated.

The GitHub Actions workflow is configured for Python 3.10 and 3.11, but protected branches and remote CI status cannot be verified until a GitHub repository exists.
