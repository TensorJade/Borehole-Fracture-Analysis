from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from borehole_fracture_analysis.connectivity import (
    calculate_connectivity,
    choose_suggestions,
    run_connectivity,
)


def sample_planes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "裂缝编号": 1,
                "钻孔号": 1,
                "法向量X": 0,
                "法向量Y": 0,
                "法向量Z": 1,
                "中心X": 500,
                "中心Y": 2000,
                "中心Z": 1000,
                "JRC": 5,
            },
            {
                "裂缝编号": 2,
                "钻孔号": 1,
                "法向量X": 0,
                "法向量Y": 0,
                "法向量Z": 1,
                "中心X": 500,
                "中心Y": 2000,
                "中心Z": 1050,
                "JRC": 6,
            },
            {
                "裂缝编号": 3,
                "钻孔号": 2,
                "法向量X": 0,
                "法向量Y": 0,
                "法向量Z": 1,
                "中心X": 1500,
                "中心Y": 2000,
                "中心Z": 1050,
                "JRC": 5.5,
            },
        ]
    )


def test_connectivity_and_suggestion_shapes() -> None:
    frame = sample_planes()
    connectivity = calculate_connectivity(frame)
    enriched, low, suggestions = choose_suggestions(frame, connectivity, top_n=2)

    assert len(connectivity) == 3
    assert np.all((connectivity["连通概率"] >= 0) & (connectivity["连通概率"] <= 1))
    assert "连通度" in enriched.columns
    assert not low.empty
    assert len(suggestions) == 2


def test_connectivity_validates_required_columns_and_normals() -> None:
    with pytest.raises(ValueError, match="缺少字段"):
        calculate_connectivity(pd.DataFrame({"裂缝编号": [1]}))

    frame = sample_planes()
    frame.loc[0, ["法向量X", "法向量Y", "法向量Z"]] = 0
    with pytest.raises(ValueError, match="零法向量"):
        calculate_connectivity(frame)


def test_connectivity_file_pipeline(tmp_path: Path) -> None:
    input_csv = tmp_path / "fracture_planes_normals.csv"
    sample_planes().to_csv(input_csv, index=False, encoding="utf-8-sig")

    connectivity, suggestions = run_connectivity(input_csv, tmp_path)

    assert not connectivity.empty
    assert not suggestions.empty
    assert (tmp_path / "fracture_connectivity_prob.csv").is_file()
    assert (tmp_path / "fracture_connectivity_degree.csv").is_file()
    assert (tmp_path / "suggested_borehole_locations.csv").is_file()
    assert (tmp_path / "connectivity_and_borehole_suggestions.png").is_file()
