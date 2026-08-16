import sys
from pathlib import Path

from borehole_fracture_analysis import cli
from borehole_fracture_analysis.cli import build_parser, get_training_pair_status


def test_parser_exposes_semantic_commands() -> None:
    arguments = build_parser().parse_args(["fit-sinusoids"])
    assert arguments.command == "fit-sinusoids"
    demo_arguments = build_parser().parse_args(["demo", "--demo-output", "demo-output"])
    assert demo_arguments.command == "demo"
    assert demo_arguments.demo_output == Path("demo-output")


def test_training_pair_status_matches_stems(tmp_path: Path, monkeypatch) -> None:
    training_path = tmp_path / "training"
    (training_path / "images").mkdir(parents=True)
    (training_path / "masks").mkdir()
    (training_path / "images" / "sample.jpg").touch()
    (training_path / "masks" / "sample_mask.png").touch()
    monkeypatch.setattr(cli, "TRAINING_DATA_PATH", training_path)

    pair_count, missing_masks, missing_images = get_training_pair_status()
    assert pair_count == 1
    assert missing_masks == []
    assert missing_images == []


def test_check_project_and_run_all_orchestration(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "raw"
    training_path = tmp_path / "training"
    model_path = tmp_path / "model.pth"
    dataset_path.mkdir()
    (dataset_path / "image.jpg").touch()
    (training_path / "images").mkdir(parents=True)
    (training_path / "masks").mkdir()
    (training_path / "images" / "sample.jpg").touch()
    (training_path / "masks" / "sample_mask.png").touch()
    model_path.touch()
    monkeypatch.setattr(cli, "DATASET_PATHS", {"sample": dataset_path})
    monkeypatch.setattr(cli, "TRAINING_DATA_PATH", training_path)

    assert cli.check_project(model_path, require_model=True)

    calls = []
    monkeypatch.setattr(cli, "ensure_artifact_directories", lambda: calls.append("ensure"))
    monkeypatch.setattr(cli, "check_project", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(cli, "run_segmentation", lambda _path: calls.append("segment"))
    monkeypatch.setattr(cli, "run_sinusoidal_stage", lambda: calls.append("sine"))
    monkeypatch.setattr(cli, "run_roughness_stage", lambda: calls.append("roughness"))
    monkeypatch.setattr(cli, "run_reconstruction_stage", lambda: calls.append("reconstruct"))
    monkeypatch.setattr(sys, "argv", ["borehole-fracture", "run-all", "--model", str(model_path)])

    cli.main()

    assert calls == ["ensure", "segment", "sine", "roughness", "reconstruct"]


def test_stage_wrappers_delegate_to_algorithm_modules(tmp_path: Path, monkeypatch) -> None:
    from borehole_fracture_analysis import (
        connectivity,
        demo,
        reconstruction,
        roughness_analysis,
        segmentation,
        sinusoidal_fitting,
    )

    model_path = tmp_path / "model.pth"
    dataset_path = tmp_path / "raw"
    model_path.touch()
    dataset_path.mkdir()
    calls = []

    monkeypatch.setattr(cli, "DATASET_PATHS", {"sample": dataset_path})
    monkeypatch.setattr(
        segmentation,
        "train_model",
        lambda *args: calls.append(("train", args)),
    )
    monkeypatch.setattr(
        segmentation,
        "predict_images_in_folder",
        lambda *args: calls.append(("segment", args)),
    )
    monkeypatch.setattr(
        sinusoidal_fitting,
        "run_sinusoidal_fitting",
        lambda *args: calls.append(("sine", args)),
    )
    monkeypatch.setattr(
        roughness_analysis,
        "run_roughness_analysis",
        lambda *args: calls.append(("roughness", args)),
    )
    monkeypatch.setattr(
        reconstruction,
        "run_reconstruction",
        lambda *args: calls.append(("reconstruct", args)),
    )
    monkeypatch.setattr(
        connectivity,
        "run_connectivity",
        lambda *args: calls.append(("connectivity", args)),
    )
    monkeypatch.setattr(
        demo,
        "run_demo",
        lambda *args: calls.append(("demo", args)),
    )

    cli.run_training(model_path, epochs=1, batch_size=1)
    cli.run_segmentation(model_path)
    cli.run_sinusoidal_stage()
    cli.run_roughness_stage()
    cli.run_reconstruction_stage()
    cli.run_demo_stage(tmp_path / "demo")

    assert [name for name, _ in calls] == [
        "train",
        "segment",
        "sine",
        "roughness",
        "reconstruct",
        "connectivity",
        "demo",
    ]
