from pathlib import Path

import cv2
import numpy as np

from borehole_fracture_analysis.reconstruction import (
    component_masks,
    estimate_jrc,
    fit_plane,
    map_component_to_3d,
    parse_depth_start,
    parse_hole_id,
    run_reconstruction,
)


def test_filename_parsers_accept_semantic_names() -> None:
    assert parse_hole_id("hole-06") == 6
    assert parse_hole_id("unknown") is None
    assert parse_depth_start("depth-02-03m_mask") == (2000.0, 1000.0)


def test_component_mapping_and_plane_fit() -> None:
    mask = np.full((20, 40), 255, dtype=np.uint8)
    mask[8:12, 2:38] = 0
    components = component_masks(mask, min_area=20, min_width_ratio=0.2)
    points = map_component_to_3d(components[0], 1, 0.0, 1000.0)
    center, normal, inclination, strike, rmse = fit_plane(points)
    raw_jrc, bounded_jrc = estimate_jrc(components[0])

    assert len(components) == 1
    assert points.shape[1] == 3
    assert center.shape == (3,)
    assert np.isclose(np.linalg.norm(normal), 1.0)
    assert 0.0 <= inclination <= 90.0
    assert 0.0 <= strike < 180.0
    assert rmse >= 0.0
    assert np.isfinite(raw_jrc)
    assert 0.0 <= bounded_jrc <= 20.0


def test_reconstruction_file_pipeline(tmp_path: Path) -> None:
    input_path = tmp_path / "masks" / "hole-01"
    output_path = tmp_path / "results"
    input_path.mkdir(parents=True)
    image = np.full((120, 240), 255, dtype=np.uint8)
    columns = np.arange(image.shape[1])
    rows = (60 + 15 * np.sin(2 * np.pi * columns / image.shape[1])).astype(int)
    for column, row in zip(columns, rows, strict=True):
        cv2.circle(image, (int(column), int(row)), 2, 0, -1)
    cv2.imwrite(str(input_path / "depth-00-01m_mask.png"), image)

    frame = run_reconstruction(input_path.parent, output_path, min_area=20, min_width_ratio=0.2)

    assert not frame.empty
    assert (output_path / "fracture_planes_normals.csv").is_file()
    assert (output_path / "fracture_plane_parameters.xlsx").is_file()
    assert (output_path / "fracture_reconstruction_3d.png").is_file()
