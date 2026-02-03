from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")

_model: CLIPModel | None = None
_processor: CLIPProcessor | None = None
_device: torch.device | None = None


def _get_model() -> Tuple[CLIPModel, CLIPProcessor, torch.device]:
    global _model, _processor, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
        _model = CLIPModel.from_pretrained(_MODEL_NAME)
        _model.to(_device)
        _model.eval()
    assert _model is not None and _processor is not None and _device is not None
    return _model, _processor, _device


def _normalize(vec: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(vec, p=2, dim=-1)


def encode_image(pil_image: Image.Image) -> np.ndarray:
    model, processor, device = _get_model()
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = _normalize(features)
    return features[0].cpu().numpy().astype(np.float32)


def encode_text(text: str) -> np.ndarray:
    model, processor, device = _get_model()
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = _normalize(features)
    return features[0].cpu().numpy().astype(np.float32)


def embedding_dim() -> int:
    model, _, _ = _get_model()
    return int(model.config.projection_dim)
