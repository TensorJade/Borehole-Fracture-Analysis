# -*- coding: utf-8 -*-
"""DBSCAN clustering and sinusoidal representation of fracture masks."""

import argparse
import logging
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN

from .config import (
    DRILL_CIRCUMFERENCE_MM,
    MASK_ARTIFACT_ROOT,
    SINUSOIDAL_ARTIFACT_PATH,
)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
LOGGER = logging.getLogger(__name__)


def imread_gray(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    return image


def sine_func(x, amplitude, period, phase, center):
    return amplitude * np.sin(2 * np.pi * x / period + phase) + center


def fit_sine(x, y, circumference, depth):
    initial = np.array(
        [
            max(0.1, (float(np.max(y)) - float(np.min(y))) / 2),
            circumference,
            0.0,
            float(np.mean(y)),
        ]
    )
    lower = np.array([0.0, circumference / 3.0, -np.pi, 0.0])
    upper = np.array([depth, circumference, np.pi, depth])
    result = least_squares(
        lambda params: sine_func(x, *params) - y,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
    )
    return result.x


def process_image(
    file_path,
    image_id,
    output_folder,
    circumference=DRILL_CIRCUMFERENCE_MM,
    depth=500.0,
    eps_mm=2.0,
    min_samples=10,
):
    image = imread_gray(file_path)
    height, width = image.shape
    coords = np.column_stack(np.where(image < 128))
    if not len(coords):
        LOGGER.warning("No fracture pixels detected: image=%s", image_id)
        return []

    y_mm = coords[:, 0] * depth / height
    x_mm = coords[:, 1] * circumference / width
    labels = DBSCAN(eps=eps_mm, min_samples=min_samples).fit_predict(np.column_stack((x_mm, y_mm)))

    results = []
    figure, axis = plt.subplots(figsize=(6, 12))
    axis.imshow(image, cmap="gray")
    for label in sorted(set(labels) - {-1}):
        xi, yi = x_mm[labels == label], y_mm[labels == label]
        x_length = float(np.ptp(xi))
        if x_length < circumference / 3.0 or len(xi) < min_samples:
            continue
        try:
            amplitude, period, phase, center = fit_sine(xi, yi, circumference, depth)
        except Exception as exc:
            LOGGER.warning("Sine fit failed: image=%s cluster=%s error=%s", image_id, label, exc)
            continue
        residual = yi - sine_func(xi, amplitude, period, phase, center)
        rmse = float(np.sqrt(np.mean(residual**2)))
        results.append(
            {
                "图像编号": image_id,
                "裂隙编号": int(label),
                "振幅R(mm)": float(amplitude),
                "周期P(mm)": float(period),
                "相位beta(rad)": float(phase),
                "中心线位置C(mm)": float(depth - center),
                "拟合RMSE(mm)": rmse,
                "像素点数": int(len(xi)),
            }
        )
        x_fit = np.linspace(0.0, circumference, 1000)
        y_fit = sine_func(x_fit, amplitude, period, phase, center)
        axis.plot(x_fit * width / circumference, y_fit * height / depth, linewidth=1.2)

    axis.set_title(f"{image_id} 裂隙正弦拟合")
    axis.axis("off")
    figure.tight_layout()
    output_folder.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_folder / f"{image_id}_fitted.png", dpi=200)
    plt.close(figure)
    return results


def run_sinusoidal_fitting(
    input_folder=MASK_ARTIFACT_ROOT / "sinusoidal_fitting",
    output_folder=SINUSOIDAL_ARTIFACT_PATH,
    depth=500.0,
    eps_mm=2.0,
    min_samples=10,
):
    """Fit sinusoidal curves to every fracture cluster in predicted masks."""
    input_folder, output_folder = Path(input_folder), Path(output_folder)
    if not input_folder.is_dir():
        raise FileNotFoundError(f"Sinusoidal mask directory does not exist: {input_folder}")
    files = sorted(p for p in input_folder.rglob("*.png") if p.is_file())
    if not files:
        raise RuntimeError(f"Sinusoidal mask directory contains no PNG files: {input_folder}")
    all_results = []
    for path in files:
        all_results.extend(
            process_image(
                path,
                path.stem.replace("_mask", ""),
                output_folder,
                depth=depth,
                eps_mm=eps_mm,
                min_samples=min_samples,
            )
        )
    if not all_results:
        raise RuntimeError("No fit candidates found; inspect masks or DBSCAN parameters")
    frame = pd.DataFrame(all_results).sort_values(["图像编号", "裂隙编号"])
    output_folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_folder / "sinusoidal_fit_results.csv", index=False, encoding="utf-8-sig")
    frame.to_excel(output_folder / "sinusoidal_fit_results.xlsx", index=False)
    LOGGER.info("Sinusoidal fitting complete: rows=%s output=%s", len(frame), output_folder)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Fracture clustering and sinusoidal fitting")
    parser.add_argument(
        "--input",
        type=Path,
        default=MASK_ARTIFACT_ROOT / "sinusoidal_fitting",
    )
    parser.add_argument("--output", type=Path, default=SINUSOIDAL_ARTIFACT_PATH)
    parser.add_argument("--depth", type=float, default=500.0)
    parser.add_argument("--eps-mm", type=float, default=2.0)
    parser.add_argument("--min-samples", type=int, default=10)
    args = parser.parse_args()
    run_sinusoidal_fitting(args.input, args.output, args.depth, args.eps_mm, args.min_samples)


if __name__ == "__main__":
    main()
