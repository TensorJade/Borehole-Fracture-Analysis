# -*- coding: utf-8 -*-
"""Reconstruct fracture components as fitted planes in three dimensions."""

import argparse
import logging
import re
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    BOREHOLES,
    DRILL_CIRCUMFERENCE_MM,
    DRILL_RADIUS_MM,
    MASK_ARTIFACT_ROOT,
    RECONSTRUCTION_ARTIFACT_PATH,
)

LOGGER = logging.getLogger(__name__)


def imread_gray(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    return image


def parse_hole_id(name):
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else None


def parse_depth_start(stem):
    match = re.search(r"(\d+)\s*-\s*(\d+)m", stem.replace("_mask", ""))
    if not match:
        raise ValueError(f"无法从文件名解析深度区间：{stem}")
    start = float(match.group(1))
    end = float(match.group(2))
    return start * 1000.0, (end - start) * 1000.0


def component_masks(mask, min_area=200, min_width_ratio=0.20):
    foreground = (mask < 128).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    result = []
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        if area >= min_area and width >= mask.shape[1] * min_width_ratio:
            result.append(labels == label_id)
    return result


def map_component_to_3d(component, hole_id, depth_start, depth_span, max_points=5000):
    rows, cols = np.where(component)
    if len(rows) > max_points:
        keep = np.linspace(0, len(rows) - 1, max_points).astype(int)
        rows, cols = rows[keep], cols[keep]
    height, width = component.shape
    theta = cols / float(width) * 2.0 * np.pi
    center_x, center_y, _ = BOREHOLES[hole_id]["position"]
    x = center_x + DRILL_RADIUS_MM * np.cos(theta)
    y = center_y + DRILL_RADIUS_MM * np.sin(theta)
    z = depth_start + rows / float(height) * depth_span
    return np.column_stack([x, y, z])


def fit_plane(points):
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    residuals = np.abs((points - center) @ normal)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    inclination = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))
    strike = float(np.degrees(np.arctan2(normal[1], normal[0])) % 180.0)
    return center, normal, inclination, strike, rmse


def estimate_jrc(component):
    rows, cols = np.where(component)
    unique_cols = np.unique(cols)
    if len(unique_cols) < 4:
        return np.nan, np.nan
    center_rows = np.array([np.median(rows[cols == col]) for col in unique_cols], dtype=float)
    x = unique_cols / component.shape[1] * DRILL_CIRCUMFERENCE_MM
    y = center_rows / component.shape[0] * 1000.0
    omega = 2.0 * np.pi / DRILL_CIRCUMFERENCE_MM
    design = np.column_stack([np.sin(omega * x), np.cos(omega * x), np.ones_like(x)])
    fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - fitted
    dx = np.maximum(np.diff(x), 1e-12)
    z2 = float(np.sqrt(np.mean((np.diff(residual) / dx) ** 2)))
    raw_jrc = float(51.85 * (z2**0.6) - 10.37)
    return raw_jrc, float(np.clip(raw_jrc, 0.0, 20.0))


def run_reconstruction(
    input_folder=MASK_ARTIFACT_ROOT / "borehole_reconstruction",
    output_folder=RECONSTRUCTION_ARTIFACT_PATH,
    min_area=200,
    min_width_ratio=0.20,
):
    """Fit three-dimensional planes to fracture components by borehole and depth."""
    input_folder, output_folder = Path(input_folder), Path(output_folder)
    if not input_folder.is_dir():
        raise FileNotFoundError(f"Borehole reconstruction mask directory missing: {input_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)
    records = []
    global_id = 1
    for hole_dir in sorted(p for p in input_folder.iterdir() if p.is_dir()):
        hole_id = parse_hole_id(hole_dir.name)
        if hole_id not in BOREHOLES:
            continue
        for image_path in sorted(hole_dir.glob("*_mask.png")):
            depth_start, depth_span = parse_depth_start(image_path.stem)
            mask = imread_gray(image_path)
            for local_id, component in enumerate(
                component_masks(mask, min_area, min_width_ratio), start=1
            ):
                points = map_component_to_3d(component, hole_id, depth_start, depth_span)
                if len(points) < 3:
                    continue
                center, normal, inclination, strike, rmse = fit_plane(points)
                jrc_raw, jrc = estimate_jrc(component)
                records.append(
                    {
                        "裂缝编号": global_id,
                        "钻孔号": hole_id,
                        "分段图像": image_path.stem.replace("_mask", ""),
                        "分段内编号": local_id,
                        "法向量X": float(normal[0]),
                        "法向量Y": float(normal[1]),
                        "法向量Z": float(normal[2]),
                        "倾角(°)": inclination,
                        "走向角(°)": strike,
                        "中心X": float(center[0]),
                        "中心Y": float(center[1]),
                        "中心Z": float(center[2]),
                        "JRC原始值": jrc_raw,
                        "JRC": jrc,
                        "平面拟合RMSE(mm)": rmse,
                        "采样点数": int(len(points)),
                    }
                )
                global_id += 1
    if not records:
        raise RuntimeError("No fracture components passed reconstruction thresholds")

    frame = pd.DataFrame(records)
    csv_path = output_folder / "fracture_planes_normals.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    frame.to_excel(output_folder / "fracture_plane_parameters.xlsx", index=False)
    render_3d(frame, output_folder / "fracture_reconstruction_3d.png")
    LOGGER.info("Reconstruction complete: planes=%s output=%s", len(frame), csv_path)
    return frame


def render_3d(frame, output_path):
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    for hole_id, info in BOREHOLES.items():
        x, y, _ = info["position"]
        axis.plot([x, x], [y, y], [0, info["depth"]], color="black", linewidth=1)
        axis.text(x, y, 0, f"{hole_id}#")
    scatter = axis.scatter(
        frame["中心X"],
        frame["中心Y"],
        frame["中心Z"],
        c=frame["JRC"],
        cmap="viridis",
        s=45,
    )
    figure.colorbar(scatter, ax=axis, shrink=0.6, label="JRC")
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Depth (mm)")
    axis.invert_zaxis()
    axis.set_title("3D borehole fracture reconstruction")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Three-dimensional fracture reconstruction")
    parser.add_argument(
        "--input",
        type=Path,
        default=MASK_ARTIFACT_ROOT / "borehole_reconstruction",
    )
    parser.add_argument("--output", type=Path, default=RECONSTRUCTION_ARTIFACT_PATH)
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--min-width-ratio", type=float, default=0.20)
    args = parser.parse_args()
    run_reconstruction(args.input, args.output, args.min_area, args.min_width_ratio)


if __name__ == "__main__":
    main()
