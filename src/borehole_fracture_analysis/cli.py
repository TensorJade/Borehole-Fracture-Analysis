"""Command-line orchestration for the complete analysis workflow."""

import argparse
import logging
from pathlib import Path

from .config import (
    ARTIFACT_ROOT,
    DATASET_PATHS,
    MASK_ARTIFACT_ROOT,
    MODEL_PATH,
    RECONSTRUCTION_ARTIFACT_PATH,
    ROUGHNESS_ARTIFACT_PATH,
    SINUSOIDAL_ARTIFACT_PATH,
    TRAINING_DATA_PATH,
    ensure_artifact_directories,
)

LOGGER = logging.getLogger(__name__)


def configure_logging(log_level: str = "INFO") -> None:
    """Configure process-wide console logging.

    Args:
        log_level: Standard Python logging level name.
    """

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_training_pair_status() -> tuple[int, list[str], list[str]]:
    """Return matched-pair count and unmatched training sample identifiers."""

    image_names = {
        image_path.stem
        for image_path in (TRAINING_DATA_PATH / "images").glob("*")
        if image_path.is_file()
    }
    mask_names = {
        mask_path.stem[:-5] if mask_path.stem.endswith("_mask") else mask_path.stem
        for mask_path in (TRAINING_DATA_PATH / "masks").glob("*")
        if mask_path.is_file()
    }
    return (
        len(image_names & mask_names),
        sorted(image_names - mask_names),
        sorted(mask_names - image_names),
    )


def check_project(model_path: Path = MODEL_PATH, require_model: bool = False) -> bool:
    """Validate local data and model prerequisites without changing files."""

    checks = []
    for dataset_name, dataset_path in DATASET_PATHS.items():
        file_count = sum(1 for path in dataset_path.rglob("*") if path.is_file())
        checks.append(
            (dataset_name, dataset_path.is_dir() and file_count > 0, f"{file_count} files")
        )

    pair_count, missing_masks, missing_images = get_training_pair_status()
    checks.append(
        (
            "training_data",
            pair_count > 0 and not missing_masks and not missing_images,
            f"{pair_count} pairs; missing masks={len(missing_masks)}; "
            f"missing images={len(missing_images)}",
        )
    )
    model_path = Path(model_path)
    checks.append(("model_checkpoint", model_path.is_file(), str(model_path)))
    for check_name, is_valid, details in checks:
        LOGGER.info("[%s] %s: %s", "OK" if is_valid else "MISSING", check_name, details)

    required_checks = checks[:-1]
    if require_model:
        required_checks = checks
    return all(is_valid for _, is_valid, _ in required_checks)


def run_training(model_path: Path, epochs: int, batch_size: int) -> None:
    """Train the segmentation model from the local paired dataset."""

    from .segmentation import train_model

    train_model(TRAINING_DATA_PATH, model_path, epochs, batch_size)


def run_segmentation(model_path: Path) -> None:
    """Generate masks for all four raw datasets."""

    from .segmentation import predict_images_in_folder

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing model checkpoint: {model_path}. Run the train command first."
        )
    for dataset_name, dataset_path in DATASET_PATHS.items():
        LOGGER.info("Segmenting dataset=%s source=%s", dataset_name, dataset_path)
        predict_images_in_folder(
            dataset_path,
            MASK_ARTIFACT_ROOT / dataset_name,
            model_path,
        )


def run_sinusoidal_stage() -> None:
    """Fit sinusoidal representations to the second dataset masks."""

    from .sinusoidal_fitting import run_sinusoidal_fitting

    run_sinusoidal_fitting(
        MASK_ARTIFACT_ROOT / "sinusoidal_fitting",
        SINUSOIDAL_ARTIFACT_PATH,
    )


def run_roughness_stage() -> None:
    """Calculate JRC values for the third dataset masks."""

    from .roughness_analysis import run_roughness_analysis

    run_roughness_analysis(
        MASK_ARTIFACT_ROOT / "roughness_analysis",
        ROUGHNESS_ARTIFACT_PATH,
    )


def run_reconstruction_stage() -> None:
    """Reconstruct planes, calculate connectivity, and suggest boreholes."""

    from .connectivity import run_connectivity
    from .reconstruction import run_reconstruction

    run_reconstruction(
        MASK_ARTIFACT_ROOT / "borehole_reconstruction",
        RECONSTRUCTION_ARTIFACT_PATH,
    )
    run_connectivity(
        RECONSTRUCTION_ARTIFACT_PATH / "fracture_planes_normals.csv",
        RECONSTRUCTION_ARTIFACT_PATH,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the public command-line parser."""

    parser = argparse.ArgumentParser(
        prog="borehole-fracture",
        description="Borehole image fracture analysis workflow",
    )
    parser.add_argument(
        "command",
        choices=(
            "check",
            "train",
            "segment",
            "fit-sinusoids",
            "analyze-roughness",
            "reconstruct",
            "run-all",
        ),
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARN", "ERROR"), default="INFO")
    return parser


def main() -> None:
    """Execute the selected workflow command."""

    arguments = build_parser().parse_args()
    configure_logging(arguments.log_level)
    ensure_artifact_directories()
    model_path = arguments.model.resolve()

    if arguments.command == "check":
        raise SystemExit(0 if check_project(model_path) else 1)
    if arguments.command == "train":
        run_training(model_path, arguments.epochs, arguments.batch_size)
        return
    if arguments.command == "segment":
        run_segmentation(model_path)
        return
    if arguments.command == "fit-sinusoids":
        run_sinusoidal_stage()
        return
    if arguments.command == "analyze-roughness":
        run_roughness_stage()
        return
    if arguments.command == "reconstruct":
        run_reconstruction_stage()
        return

    if not check_project(model_path):
        raise RuntimeError("Raw datasets or training data are incomplete")
    if not model_path.is_file():
        if not arguments.train_if_missing:
            raise FileNotFoundError(
                "Model checkpoint is missing. Run `borehole-fracture train` first."
            )
        run_training(model_path, arguments.epochs, arguments.batch_size)
    run_segmentation(model_path)
    run_sinusoidal_stage()
    run_roughness_stage()
    run_reconstruction_stage()
    LOGGER.info("Workflow complete: %s", ARTIFACT_ROOT)


if __name__ == "__main__":
    main()
