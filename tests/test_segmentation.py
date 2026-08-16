from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from borehole_fracture_analysis.segmentation import (
    AttentionUNet,
    CrackDataset,
    predict_images_in_folder,
    segmentation_loss,
    train_model,
)


def test_crack_dataset_pairs_image_and_mask(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    Image.fromarray(np.full((12, 10, 3), 127, dtype=np.uint8)).save(image_dir / "sample.jpg")
    Image.fromarray(np.full((12, 10), 255, dtype=np.uint8)).save(mask_dir / "sample_mask.png")

    dataset = CrackDataset(tmp_path, resize=(16, 16))
    image, mask = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (3, 16, 16)
    assert mask.shape == (1, 16, 16)
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_crack_dataset_rejects_missing_directories(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CrackDataset(tmp_path)


def test_segmentation_loss_is_finite_and_differentiable() -> None:
    prediction = torch.full((2, 1, 8, 8), 0.6, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[:, :, 2:6, 2:6] = 1.0

    loss = segmentation_loss(prediction, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None


def test_attention_unet_preserves_spatial_shape() -> None:
    model = AttentionUNet().eval()
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 32, 32)))

    assert output.shape == (1, 1, 32, 32)
    assert torch.all((0.0 <= output) & (output <= 1.0))


class TinySegmentationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = torch.nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.output(tensor))


def test_train_and_predict_file_pipeline(tmp_path: Path, monkeypatch) -> None:
    from borehole_fracture_analysis import segmentation

    training_path = tmp_path / "training"
    image_dir = training_path / "images"
    mask_dir = training_path / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir()
    for index in range(2):
        image = np.full((24, 24, 3), 80 + index * 40, dtype=np.uint8)
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[:, 10:14] = 255
        Image.fromarray(image).save(image_dir / f"sample-{index}.png")
        Image.fromarray(mask).save(mask_dir / f"sample-{index}_mask.png")

    artifact_path = tmp_path / "artifacts"
    checkpoint = tmp_path / "tiny.pth"
    monkeypatch.setattr(segmentation, "AttentionUNet", TinySegmentationModel)
    monkeypatch.setattr(segmentation, "SEGMENTATION_ARTIFACT_PATH", artifact_path)
    monkeypatch.setattr(segmentation, "ensure_artifact_directories", lambda: None)

    trained_path = train_model(
        training_path,
        checkpoint,
        epochs=1,
        batch_size=1,
        val_ratio=0.5,
        seed=7,
    )
    outputs = predict_images_in_folder(image_dir, tmp_path / "predicted", checkpoint)

    assert trained_path == checkpoint
    assert checkpoint.is_file()
    assert (artifact_path / "training_history.json").is_file()
    assert len(outputs) == 2
    assert all(path.is_file() for path in outputs)
