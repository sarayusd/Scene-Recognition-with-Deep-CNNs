from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.config import MODEL_PATH
from src.models.student import EfficientSceneStudent
from src.transforms import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    build_inference_transform,
)


@dataclass(frozen=True)
class Prediction:
    class_name: str
    confidence: float


class SceneClassifier:
    """Load the Indoor-67 model and perform image classification."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {self.model_path}"
            )

        self.device = self._resolve_device(device)
        bundle = self._load_bundle()

        self.class_names = self._validate_class_names(
            bundle.get("class_names")
        )

        self.image_size = int(bundle.get("image_size", 224))
        self.mean = bundle.get("mean", DEFAULT_MEAN)
        self.std = bundle.get("std", DEFAULT_STD)

        self.transform = build_inference_transform(
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
        )

        self.model = EfficientSceneStudent(
            num_classes=len(self.class_names)
        )

        try:
            self.model.load_state_dict(
                bundle["model_state_dict"],
                strict=True,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "The checkpoint weights do not match "
                "src/models/student.py. Ensure the model architecture "
                "exactly matches the training notebook."
            ) from exc

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(
        requested_device: str | None,
    ) -> torch.device:
        if requested_device:
            return torch.device(requested_device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    def _load_bundle(self) -> dict[str, Any]:
        try:
            bundle = torch.load(
                self.model_path,
                map_location=self.device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load model artifact: {self.model_path}"
            ) from exc

        if not isinstance(bundle, dict):
            raise TypeError(
                "Expected the checkpoint to contain a dictionary"
            )

        if "model_state_dict" not in bundle:
            raise KeyError(
                "Checkpoint is missing 'model_state_dict'"
            )

        return bundle

    @staticmethod
    def _validate_class_names(
        class_names: Any,
    ) -> list[str]:
        if not isinstance(class_names, (list, tuple)):
            raise TypeError(
                "Checkpoint must contain class_names as a list or tuple"
            )

        names = [str(name) for name in class_names]

        if not names:
            raise ValueError("class_names cannot be empty")

        return names

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        top_k: int = 5,
    ) -> list[Prediction]:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        image = image.convert("RGB")

        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)

        top_k = min(top_k, len(self.class_names))

        confidences, indices = torch.topk(
            probabilities,
            k=top_k,
            dim=1,
        )

        predictions = []

        for confidence, index in zip(
            confidences[0].cpu().tolist(),
            indices[0].cpu().tolist(),
        ):
            predictions.append(
                Prediction(
                    class_name=self.class_names[index],
                    confidence=float(confidence),
                )
            )

        return predictions