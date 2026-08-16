"""Self-contained demonstration model and synthetic borehole image workflow."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import ARTIFACT_ROOT

LOGGER = logging.getLogger(__name__)
DEMO_MODEL_PARAMETERS_PATH = (
    Path(__file__).resolve().parent / "resources" / "demo_linear_segmenter.json"
)


@dataclass(frozen=True)
class DemoModelParameters:
    """Validated parameters for the lightweight RGB pixel classifier.

    Attributes:
        channel_weights: Linear weights applied to RGB values scaled to ``[0, 1]``.
        bias: Linear-classifier bias added before the sigmoid function.
        probability_threshold: Minimum fracture probability used for classification.
        morphology_kernel: Odd closing-kernel width used to join nearby fracture pixels.
        min_component_area: Connected components smaller than this area are discarded.
    """

    channel_weights: tuple[float, float, float]
    bias: float
    probability_threshold: float
    morphology_kernel: int
    min_component_area: int

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DemoModelParameters":
        """Build and validate model parameters loaded from JSON.

        Args:
            payload: Parsed JSON object containing the ``parameters`` mapping.

        Returns:
            Validated immutable model parameters.

        Raises:
            ValueError: If a required value is missing or outside its valid range.
        """

        try:
            parameters = payload["parameters"]
            weights = tuple(float(value) for value in parameters["channel_weights"])
            bias = float(parameters["bias"])
            threshold = float(parameters["probability_threshold"])
            kernel = int(parameters["morphology_kernel"])
            min_area = int(parameters["min_component_area"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Demo model JSON contains invalid or missing parameters") from exc

        if len(weights) != 3:
            raise ValueError("channel_weights must contain exactly three RGB values")
        if not 0.0 < threshold < 1.0:
            raise ValueError("probability_threshold must be between zero and one")
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("morphology_kernel must be a positive odd integer")
        if min_area < 0:
            raise ValueError("min_component_area cannot be negative")
        return cls(weights, bias, threshold, kernel, min_area)


class DemoLinearPixelModel:
    """Small deterministic classifier that assigns fracture probability per RGB pixel."""

    def __init__(self, parameters: DemoModelParameters):
        self.parameters = parameters

    def predict_probability(self, image: np.ndarray) -> np.ndarray:
        """Calculate a fracture-probability map for one RGB image.

        Args:
            image: Unsigned 8-bit RGB array with shape ``(height, width, 3)``.

        Returns:
            Float probability array with shape ``(height, width)``.

        Raises:
            ValueError: If the image does not satisfy the RGB input contract.
        """

        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Demo model input must be an unsigned 8-bit RGB image")
        normalized = image.astype(np.float32) / 255.0
        weights = np.asarray(self.parameters.channel_weights, dtype=np.float32)
        logits = normalized @ weights + self.parameters.bias
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def load_demo_model(
    parameters_path: Path = DEMO_MODEL_PARAMETERS_PATH,
) -> DemoLinearPixelModel:
    """Load the distributable demonstration model from its JSON parameters.

    Args:
        parameters_path: JSON model-parameter file included with the package.

    Returns:
        Ready-to-run deterministic demonstration model.

    Raises:
        FileNotFoundError: If the parameter file does not exist.
        ValueError: If the JSON document or model parameters are invalid.
    """

    parameters_path = Path(parameters_path)
    try:
        payload = json.loads(parameters_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid demo model JSON: {parameters_path}") from exc
    return DemoLinearPixelModel(DemoModelParameters.from_mapping(payload))


def generate_synthetic_borehole_image(
    output_path: Path,
    width: int = 640,
    height: int = 320,
    seed: int = 42,
) -> Path:
    """Generate a deterministic unwrapped-borehole image with a dark sinusoidal fracture.

    Args:
        output_path: Destination PNG path.
        width: Generated image width in pixels.
        height: Generated image height in pixels.
        seed: Random seed used for reproducible texture noise.

    Returns:
        The generated image path.

    Raises:
        ValueError: If the requested image dimensions are too small.
    """

    if width < 64 or height < 64:
        raise ValueError("Demo image width and height must both be at least 64 pixels")
    random_generator = np.random.default_rng(seed)
    x_coordinates = np.arange(width, dtype=np.float32)
    y_coordinates = np.arange(height, dtype=np.float32)[:, None]
    background = (
        205.0
        + 14.0 * np.sin(2.0 * np.pi * x_coordinates / 83.0)[None, :]
        + 8.0 * np.cos(2.0 * np.pi * y_coordinates / 57.0)
    )
    noise = random_generator.normal(0.0, 4.0, size=(height, width))
    grayscale = np.clip(background + noise, 0.0, 255.0).astype(np.uint8)
    image = np.repeat(grayscale[:, :, None], 3, axis=2)

    fracture_rows = (
        height * 0.5 + height * 0.16 * np.sin(2.0 * np.pi * x_coordinates / width + 0.35)
    ).astype(np.int32)
    fracture_points = np.column_stack((x_coordinates.astype(np.int32), fracture_rows))
    cv2.polylines(image, [fracture_points], False, (24, 24, 24), thickness=4)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)
    return output_path


def _remove_small_components(foreground: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return foreground
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    filtered = np.zeros_like(foreground, dtype=bool)
    for component_id in range(1, component_count):
        if statistics[component_id, cv2.CC_STAT_AREA] >= min_area:
            filtered |= labels == component_id
    return filtered


def predict_demo_image(
    image_path: Path,
    mask_path: Path,
    overlay_path: Path,
    parameters_path: Path = DEMO_MODEL_PARAMETERS_PATH,
) -> dict[str, Any]:
    """Run the demo model and save a binary mask plus a red visual overlay.

    Args:
        image_path: Input RGB image path.
        mask_path: Destination binary mask path; fractures are black.
        overlay_path: Destination visualization path; fractures are red.
        parameters_path: JSON parameter file for the demo model.

    Returns:
        Serializable inference metadata including output paths and pixel fraction.

    Raises:
        FileNotFoundError: If the input image or parameter file is missing.
        ValueError: If the parameter document or input image is invalid.
    """

    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Demo input image does not exist: {image_path}")
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    model = load_demo_model(parameters_path)
    probability = model.predict_probability(image)
    foreground = probability >= model.parameters.probability_threshold

    kernel_size = model.parameters.morphology_kernel
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        foreground = cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(
            bool
        )
    foreground = _remove_small_components(foreground, model.parameters.min_component_area)

    mask = np.where(foreground, 0, 255).astype(np.uint8)
    overlay = image.copy()
    overlay[foreground] = (0.30 * overlay[foreground] + 0.70 * np.array([255, 0, 0])).astype(
        np.uint8
    )
    mask_path, overlay_path = Path(mask_path), Path(overlay_path)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(mask_path)
    Image.fromarray(overlay).save(overlay_path)

    return {
        "model_type": "linear_rgb_pixel_classifier",
        "model_parameters": str(Path(parameters_path).resolve()),
        "input_image": str(image_path.resolve()),
        "mask_image": str(mask_path.resolve()),
        "overlay_image": str(overlay_path.resolve()),
        "fracture_pixel_fraction": float(np.mean(foreground)),
    }


def run_demo(
    output_directory: Path = ARTIFACT_ROOT / "demo",
    parameters_path: Path = DEMO_MODEL_PARAMETERS_PATH,
) -> dict[str, Any]:
    """Generate a synthetic input and execute the packaged demonstration model.

    Args:
        output_directory: Directory receiving the input, mask, overlay, and summary.
        parameters_path: JSON parameter file for the demo model.

    Returns:
        Serializable metadata describing the generated demonstration artifacts.
    """

    output_directory = Path(output_directory)
    input_path = generate_synthetic_borehole_image(output_directory / "synthetic_borehole.png")
    summary = predict_demo_image(
        input_path,
        output_directory / "demo_fracture_mask.png",
        output_directory / "demo_overlay.png",
        parameters_path,
    )
    summary_path = output_directory / "demo_summary.json"
    summary["summary"] = str(summary_path.resolve())
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Runnable demo complete: output=%s", output_directory.resolve())
    return summary
