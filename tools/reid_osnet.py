"""Torchreid OSNet embedding helper."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


class OSNetEmbedder:
    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "osnet_x1_0",
        checkpoint_path: str | Path | None = None,
        pretrained: bool = True,
    ) -> None:
        os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

        import torch  # type: ignore
        from torchreid.reid import models  # type: ignore

        self.torch = torch
        self.device = normalize_device(device, torch)
        self.model = models.build_model(model_name, num_classes=1000, pretrained=pretrained and checkpoint_path is None)
        if checkpoint_path is not None:
            load_checkpoint_weights(torch, self.model, resolve_project_path(checkpoint_path))
        self.model.eval()
        self.model.to(self.device)
        self.model_name = model_name

    def embed_bgr(self, cv2: Any, crop: Any) -> list[float]:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return self.embed_rgb_array(cv2, rgb)

    def embed_rgb_array(self, cv2: Any, rgb: Any) -> list[float]:
        torch = self.torch
        resized = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = (tensor - mean) / std
        with torch.no_grad():
            features = self.model(tensor)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        return [float(value) for value in features.squeeze(0).cpu().tolist()]


def normalize_device(device: str, torch: Any) -> str:
    if device in ("cuda", "cuda:0", "0") and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def load_checkpoint_weights(torch: Any, model: Any, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"ReID checkpoint not found: {path}\n"
            "Download it with: ./venv/bin/python tools/download_soccernet_reid_model.py"
        )
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        clean_key = clean_state_key(key)
        if clean_key in model_state and tuple(model_state[clean_key].shape) == tuple(value.shape):
            compatible[clean_key] = value
        else:
            skipped.append(clean_key)
    model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)} ReID tensors from {path}")
    if skipped:
        print(f"Skipped {len(skipped)} incompatible checkpoint tensors")


def extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Unsupported ReID checkpoint format")


def clean_state_key(key: str) -> str:
    for prefix in ("module.", "model.", "net."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key
