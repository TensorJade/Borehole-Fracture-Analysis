import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from borehole_fracture_analysis.demo import (
    DEMO_MODEL_PARAMETERS_PATH,
    DemoLinearPixelModel,
    DemoModelParameters,
    generate_synthetic_borehole_image,
    load_demo_model,
    predict_demo_image,
    run_demo,
)


def test_demo_model_assigns_higher_probability_to_dark_pixels() -> None:
    model = load_demo_model()
    image = np.array([[[20, 20, 20], [240, 240, 240]]], dtype=np.uint8)

    probabilities = model.predict_probability(image)

    assert DEMO_MODEL_PARAMETERS_PATH.is_file()
    assert probabilities.shape == (1, 2)
    assert probabilities[0, 0] > 0.9
    assert probabilities[0, 1] < 0.1


def test_demo_file_and_api_workflow(tmp_path: Path) -> None:
    output_directory = tmp_path / "demo"
    input_path = generate_synthetic_borehole_image(
        output_directory / "custom-input.png", width=160, height=96
    )
    custom_summary = predict_demo_image(
        input_path,
        output_directory / "custom-mask.png",
        output_directory / "custom-overlay.png",
    )
    summary = run_demo(tmp_path / "complete-demo")

    mask = np.asarray(Image.open(custom_summary["mask_image"]))
    assert set(np.unique(mask).tolist()) == {0, 255}
    assert 0.0 < custom_summary["fracture_pixel_fraction"] < 0.2
    for key in ("input_image", "mask_image", "overlay_image", "summary"):
        assert Path(summary[key]).is_file()
    loaded_summary = json.loads(Path(summary["summary"]).read_text(encoding="utf-8"))
    assert loaded_summary["model_type"] == "linear_rgb_pixel_classifier"


def test_demo_parameter_and_input_validation(tmp_path: Path) -> None:
    invalid_parameters = tmp_path / "invalid.json"
    invalid_parameters.write_text('{"parameters": {}}', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid or missing"):
        load_demo_model(invalid_parameters)
    with pytest.raises(ValueError, match="at least 64"):
        generate_synthetic_borehole_image(tmp_path / "small.png", width=32, height=32)

    model = DemoLinearPixelModel(DemoModelParameters((-4.0, -4.0, -4.0), 7.2, 0.5, 3, 10))
    with pytest.raises(ValueError, match="unsigned 8-bit RGB"):
        model.predict_probability(np.zeros((10, 10), dtype=np.uint8))
