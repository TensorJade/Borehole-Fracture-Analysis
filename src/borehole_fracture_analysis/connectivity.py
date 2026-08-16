# -*- coding: utf-8 -*-
"""Fracture connectivity, uncertainty field, and borehole suggestions."""

import argparse
import logging
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import BOREHOLES, RECONSTRUCTION_ARTIFACT_PATH

LOGGER = logging.getLogger(__name__)


DIST_THRESHOLD = 150.0
D_MIN = 50.0
ANGLE_THRESHOLD = 15.0
JRC_THRESHOLD = 5.0
UNCERTAINTY_RADIUS = 150.0
GRID_STEP = 100.0
TOP_N_HOLES = 3
MIN_SUGGESTION_SPACING = 300.0


REQUIRED_COLUMNS = {
    "裂缝编号",
    "钻孔号",
    "法向量X",
    "法向量Y",
    "法向量Z",
    "中心X",
    "中心Y",
    "中心Z",
    "JRC",
}


def calculate_connectivity(frame):
    """Calculate pairwise connectivity probabilities for reconstructed planes."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"裂隙平面表缺少字段：{sorted(missing)}")
    normals = frame[["法向量X", "法向量Y", "法向量Z"]].to_numpy(float)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("裂隙平面表包含零法向量")
    normals /= norms
    results = []
    for i, j in combinations(range(len(frame)), 2):
        row_i, row_j = frame.iloc[i], frame.iloc[j]
        center_i = row_i[["中心X", "中心Y", "中心Z"]].to_numpy(float)
        center_j = row_j[["中心X", "中心Y", "中心Z"]].to_numpy(float)
        distance = float(np.linalg.norm(center_i - center_j))
        same_hole = int(row_i["钻孔号"]) == int(row_j["钻孔号"])
        angle = np.nan
        jrc_diff = np.nan
        if same_hole:
            if distance <= D_MIN:
                probability = 1.0
            elif distance <= DIST_THRESHOLD:
                probability = (DIST_THRESHOLD - distance) / (DIST_THRESHOLD - D_MIN)
            else:
                probability = 0.0
            relation = "same"
        else:
            angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(normals[i], normals[j])), 0, 1))))
            jrc_diff = float(abs(float(row_i["JRC"]) - float(row_j["JRC"])))
            p_angle = max(0.0, 1.0 - angle / ANGLE_THRESHOLD)
            p_jrc = max(0.0, 1.0 - jrc_diff / JRC_THRESHOLD)
            probability = p_angle * p_jrc
            relation = "cross"
        results.append(
            {
                "索引1": i,
                "索引2": j,
                "裂缝1": int(row_i["裂缝编号"]),
                "裂缝2": int(row_j["裂缝编号"]),
                "类型": relation,
                "距离_mm": distance,
                "夹角_deg": angle,
                "JRC差": jrc_diff,
                "连通概率": float(probability),
            }
        )
    return pd.DataFrame(results)


def choose_suggestions(
    frame,
    connectivity,
    top_n=TOP_N_HOLES,
    radius=UNCERTAINTY_RADIUS,
    grid_step=GRID_STEP,
    min_spacing=MIN_SUGGESTION_SPACING,
):
    """Rank candidate borehole positions around low-connectivity fractures."""
    degree = np.zeros(len(frame), dtype=int)
    for row in connectivity.itertuples():
        if row.连通概率 > 0:
            degree[row.索引1] += 1
            degree[row.索引2] += 1
    frame = frame.copy()
    frame["连通度"] = degree
    low = frame[frame["连通度"] <= 1]
    if low.empty:
        low = frame.nsmallest(min(len(frame), max(1, top_n)), "连通度")
    uncertainty_xy = low[["中心X", "中心Y"]].to_numpy(float)

    hole_xy = np.array([info["position"][:2] for info in BOREHOLES.values()], dtype=float)
    xmin, ymin = hole_xy.min(axis=0) - radius
    xmax, ymax = hole_xy.max(axis=0) + radius
    x_grid = np.arange(xmin, xmax + grid_step, grid_step)
    y_grid = np.arange(ymin, ymax + grid_step, grid_step)
    grid = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)
    tree = cKDTree(uncertainty_xy)
    density = np.array([len(indices) for indices in tree.query_ball_point(grid, r=radius)])

    selected = []
    for index in np.argsort(-density, kind="stable"):
        candidate = grid[index]
        if all(np.linalg.norm(candidate - previous[0]) >= min_spacing for previous in selected):
            selected.append((candidate, int(density[index])))
        if len(selected) == top_n:
            break
    suggestions = pd.DataFrame(
        [
            {
                "优先级": rank,
                "X(mm)": point[0],
                "Y(mm)": point[1],
                "Z(mm)": 0.0,
                "不确定裂隙覆盖数": density_value,
            }
            for rank, (point, density_value) in enumerate(selected, start=1)
        ]
    )
    return frame, low, suggestions


def render_result(frame, connectivity, low, suggestions, output_path):
    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")
    for hole_id, info in BOREHOLES.items():
        x, y, _ = info["position"]
        axis.plot([x, x], [y, y], [0, info["depth"]], color="black", linewidth=1.5)
        axis.text(x, y, 0, f"{hole_id}#")
    scatter = axis.scatter(
        frame["中心X"],
        frame["中心Y"],
        frame["中心Z"],
        c=frame["连通度"],
        cmap="viridis",
        s=45,
    )
    visible_edges = connectivity[connectivity["连通概率"] >= 0.25].nlargest(200, "连通概率")
    for row in visible_edges.itertuples():
        if row.连通概率 <= 0:
            continue
        first, second = frame.iloc[row.索引1], frame.iloc[row.索引2]
        axis.plot(
            [first["中心X"], second["中心X"]],
            [first["中心Y"], second["中心Y"]],
            [first["中心Z"], second["中心Z"]],
            color="orange" if row.类型 == "cross" else "green",
            alpha=max(0.1, row.连通概率),
        )
    axis.scatter(low["中心X"], low["中心Y"], low["中心Z"], c="red", marker="x", s=70)
    if not suggestions.empty:
        axis.scatter(
            suggestions["X(mm)"],
            suggestions["Y(mm)"],
            suggestions["Z(mm)"],
            c="magenta",
            marker="^",
            s=100,
        )
    figure.colorbar(scatter, ax=axis, shrink=0.6, label="Connectivity degree")
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Depth (mm)")
    axis.invert_zaxis()
    axis.set_title("Fracture connectivity and suggested boreholes")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def run_connectivity(
    input_csv=RECONSTRUCTION_ARTIFACT_PATH / "fracture_planes_normals.csv",
    output_folder=RECONSTRUCTION_ARTIFACT_PATH,
):
    input_csv, output_folder = Path(input_csv), Path(output_folder)
    if not input_csv.is_file():
        raise FileNotFoundError(f"Missing reconstruction output: {input_csv}")
    output_folder.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_csv)
    if len(frame) < 2:
        raise RuntimeError("至少需要2个裂隙平面才能进行连通性分析")
    connectivity = calculate_connectivity(frame)
    frame, low, suggestions = choose_suggestions(frame, connectivity)
    connectivity.drop(columns=["索引1", "索引2"]).to_csv(
        output_folder / "fracture_connectivity_prob.csv", index=False, encoding="utf-8-sig"
    )
    frame.to_csv(
        output_folder / "fracture_connectivity_degree.csv",
        index=False,
        encoding="utf-8-sig",
    )
    suggestions.to_csv(
        output_folder / "suggested_borehole_locations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    render_result(
        frame,
        connectivity,
        low,
        suggestions,
        output_folder / "connectivity_and_borehole_suggestions.png",
    )
    LOGGER.info("Suggested boreholes:\n%s", suggestions.to_string(index=False))
    return connectivity, suggestions


def main():
    parser = argparse.ArgumentParser(description="Fracture connectivity and borehole suggestions")
    parser.add_argument(
        "--input",
        type=Path,
        default=RECONSTRUCTION_ARTIFACT_PATH / "fracture_planes_normals.csv",
    )
    parser.add_argument("--output", type=Path, default=RECONSTRUCTION_ARTIFACT_PATH)
    args = parser.parse_args()
    run_connectivity(args.input, args.output)


if __name__ == "__main__":
    main()
