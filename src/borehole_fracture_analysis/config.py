"""Central filesystem and borehole configuration.

Paths are derived from the installed source tree and never depend on the
process working directory. Raw data, model weights, and generated artifacts
are intentionally separated because they have different distribution rules.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw"

DATASET_PATHS = {
    "segmentation": RAW_DATA_ROOT / "segmentation-images",
    "sinusoidal_fitting": RAW_DATA_ROOT / "sinusoidal-fractures",
    "roughness_analysis": RAW_DATA_ROOT / "roughness-images",
    "borehole_reconstruction": RAW_DATA_ROOT / "borehole-scans",
}

TRAINING_DATA_PATH = DATA_ROOT / "training"
EXAMPLE_DATA_PATH = DATA_ROOT / "examples"
MODEL_PATH = PROJECT_ROOT / "models" / "au_net_crack.pth"

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
MASK_ARTIFACT_ROOT = ARTIFACT_ROOT / "masks"
SEGMENTATION_ARTIFACT_PATH = ARTIFACT_ROOT / "segmentation"
SINUSOIDAL_ARTIFACT_PATH = ARTIFACT_ROOT / "sinusoidal-fitting"
ROUGHNESS_ARTIFACT_PATH = ARTIFACT_ROOT / "roughness-analysis"
RECONSTRUCTION_ARTIFACT_PATH = ARTIFACT_ROOT / "reconstruction"

BOREHOLES = {
    1: {"position": (500.0, 2000.0, 0.0), "depth": 7000.0},
    2: {"position": (1500.0, 2000.0, 0.0), "depth": 7000.0},
    3: {"position": (2500.0, 2000.0, 0.0), "depth": 7000.0},
    4: {"position": (500.0, 1000.0, 0.0), "depth": 5000.0},
    5: {"position": (1500.0, 1000.0, 0.0), "depth": 7000.0},
    6: {"position": (2500.0, 1000.0, 0.0), "depth": 7000.0},
}

DRILL_CIRCUMFERENCE_MM = 94.25
DRILL_RADIUS_MM = DRILL_CIRCUMFERENCE_MM / (2.0 * 3.141592653589793)


def ensure_artifact_directories() -> None:
    """Create generated-artifact directories if they do not already exist."""

    artifact_paths = (
        ARTIFACT_ROOT,
        MASK_ARTIFACT_ROOT,
        SEGMENTATION_ARTIFACT_PATH,
        SINUSOIDAL_ARTIFACT_PATH,
        ROUGHNESS_ARTIFACT_PATH,
        RECONSTRUCTION_ARTIFACT_PATH,
    )
    for artifact_path in artifact_paths:
        artifact_path.mkdir(parents=True, exist_ok=True)
