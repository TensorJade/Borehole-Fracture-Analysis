from pathlib import Path

import cv2
import numpy as np

from borehole_fracture_analysis.sinusoidal_fitting import (
    fit_sine,
    process_image,
    run_sinusoidal_fitting,
    sine_func,
)


def test_sine_function_and_robust_fit_recover_curve() -> None:
    x = np.linspace(0.0, 94.25, 300)
    expected = np.array([12.0, 94.25, 0.35, 250.0])
    y = sine_func(x, *expected)

    fitted = fit_sine(x, y, circumference=94.25, depth=500.0)
    predicted = sine_func(x, *fitted)

    assert np.sqrt(np.mean((predicted - y) ** 2)) < 1e-3
    assert np.allclose(sine_func(np.array([0.0]), 2.0, 10.0, 0.0, 3.0), [3.0])


def test_sinusoidal_file_pipeline(tmp_path: Path) -> None:
    input_path = tmp_path / "masks"
    output_path = tmp_path / "results"
    input_path.mkdir()
    image = np.full((240, 400), 255, dtype=np.uint8)
    columns = np.arange(image.shape[1])
    rows = (120 + 30 * np.sin(2 * np.pi * columns / image.shape[1])).astype(int)
    for column, row in zip(columns, rows, strict=True):
        cv2.circle(image, (int(column), int(row)), 2, 0, -1)
    cv2.imwrite(str(input_path / "sinusoidal-01_mask.png"), image)

    direct_rows = process_image(
        input_path / "sinusoidal-01_mask.png",
        "sinusoidal-01",
        output_path,
        eps_mm=3.0,
        min_samples=3,
    )
    frame = run_sinusoidal_fitting(
        input_path,
        output_path,
        eps_mm=3.0,
        min_samples=3,
    )

    assert direct_rows
    assert not frame.empty
    assert (output_path / "sinusoidal_fit_results.csv").is_file()
    assert (output_path / "sinusoidal_fit_results.xlsx").is_file()
