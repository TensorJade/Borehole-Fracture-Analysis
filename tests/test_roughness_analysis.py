from pathlib import Path

import cv2
import numpy as np

from borehole_fracture_analysis.roughness_analysis import (
    _uniform_sample_from_base,
    adapt_preprocess_for_range,
    arclength,
    compute_Z2_JRC,
    curvature_adaptive_sampling,
    estimate_curvature,
    gradient_adaptive_sampling,
    img_coords_to_mm,
    rdp,
    resample_equal_arclength,
    run_roughness_analysis,
    truncate_JRC,
)


def test_coordinate_conversion_and_arclength() -> None:
    x, y, dx, dy = img_coords_to_mm([0, 100], [0, 50], 100, 200)

    assert np.allclose(x, [0.0, 47.125])
    assert np.allclose(y, [495.0, 245.0])
    assert dx == 94.25 / 200
    assert dy == 5.0
    assert np.allclose(arclength([0, 3], [0, 4]), [0, 5])


def test_sampling_helpers_preserve_endpoints() -> None:
    x = np.linspace(0.0, 20.0, 101)
    y = np.sin(x)

    xr, yr = resample_equal_arclength(x, y, 0.5)
    xc, yc = curvature_adaptive_sampling(x, y, n_target=20)
    xg, yg = gradient_adaptive_sampling(x, y, n_target=20)
    xu, yu = _uniform_sample_from_base(x, y, 0.4, 0.2)
    xd, yd = rdp(x, y, eps_mm=0.1)

    assert len(xr) > 2 and len(xr) == len(yr)
    for sampled_x, sampled_y in ((xc, yc), (xg, yg), (xu, yu), (xd, yd)):
        assert sampled_x[0] == x[0]
        assert sampled_x[-1] == x[-1]
        assert len(sampled_x) == len(sampled_y)
    assert np.all(estimate_curvature(x, np.zeros_like(x)) == 0)


def test_jrc_calculation_and_range_adaptation() -> None:
    x = np.linspace(0.0, 100.0, 401)
    y = 0.05 * np.sin(x)
    z2, jrc = compute_Z2_JRC(x, y)
    result = adapt_preprocess_for_range(x, y)

    assert z2 > 0
    assert np.isfinite(jrc)
    assert len(result) == 7
    assert truncate_JRC(-2.0) == 0.0
    assert truncate_JRC(25.0) == 20.0


def test_roughness_file_pipeline(tmp_path: Path) -> None:
    input_path = tmp_path / "masks"
    output_path = tmp_path / "results"
    input_path.mkdir()
    image = np.full((240, 400), 255, dtype=np.uint8)
    columns = np.arange(image.shape[1])
    rows = (
        120
        + 25 * np.sin(2 * np.pi * columns / image.shape[1])
        + 3 * np.sin(2 * np.pi * columns / 17)
    ).astype(int)
    for column, row in zip(columns, rows, strict=True):
        cv2.circle(image, (int(column), int(row)), 1, 0, -1)
    cv2.imwrite(str(input_path / "roughness-01_mask.png"), image)

    frame = run_roughness_analysis(input_path, output_path)

    assert not frame.empty
    assert (output_path / "roughness_jrc_results.csv").is_file()
    assert (output_path / "roughness_jrc_results.xlsx").is_file()
