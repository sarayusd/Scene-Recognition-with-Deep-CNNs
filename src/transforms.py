from __future__ import annotations

from collections.abc import Sequence

from torchvision import transforms


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def build_inference_transform(
    image_size: int = 224,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> transforms.Compose:
    """Build deterministic preprocessing for inference."""

    if image_size <= 0:
        raise ValueError("image_size must be greater than zero")

    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must each contain three values")

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )