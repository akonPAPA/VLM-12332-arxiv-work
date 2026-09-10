"""Configuration dataclasses — single source of truth for all dimensions.

Configs are plain dataclasses loadable from YAML. Nothing here imports torch,
so it stays cheap to introspect (e.g. from param-counting scripts).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import yaml


@dataclass
class VisionConfig:
    """ViT image encoder."""

    image_size: int = 448
    patch_size: int = 14
    dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 8 / 3  # SwiGLU hidden = mlp_ratio * dim
    in_channels: int = 3
    qk_norm: bool = True
    norm_eps: float = 1e-5

    @property
    def num_patches(self) -> int:
        side = self.image_size // self.patch_size
        return side * side

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size {self.image_size} must be divisible by "
                f"patch_size {self.patch_size}"
            )
        if self.dim % self.num_heads != 0:
            raise ValueError("vision dim must be divisible by num_heads")


@dataclass
class ResamplerConfig:
    """Perceiver Resampler: patches -> fixed number of visual tokens."""

    num_latents: int = 64
    depth: int = 6
    num_heads: int = 16
    mlp_ratio: float = 8 / 3
    qk_norm: bool = True
    norm_eps: float = 1e-5
    # dim is taken from VisionConfig.dim (resampler works in vision width)


@dataclass
class TextConfig:
    """Decoder-only text model with gated cross-attention to visual tokens."""

    vocab_size: int = 151936  # placeholder; overwritten from the real tokenizer
    dim: int = 2048
    depth: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16  # set < num_heads for GQA
    mlp_ratio: float = 8 / 3
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    qk_norm: bool = True
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.dim % self.num_heads != 0:
            raise ValueError("text dim must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")


@dataclass
class VLMConfig:
    """Top-level config binding vision, resampler and text."""

    vision: VisionConfig = field(default_factory=VisionConfig)
    resampler: ResamplerConfig = field(default_factory=ResamplerConfig)
    text: TextConfig = field(default_factory=TextConfig)
    gradient_checkpointing: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VLMConfig":
        d = dict(d)
        vision = VisionConfig(**d.pop("vision", {}))
        resampler = ResamplerConfig(**d.pop("resampler", {}))
        text = TextConfig(**d.pop("text", {}))
        return cls(vision=vision, resampler=resampler, text=text, **d)

    @classmethod
    def from_yaml(cls, path: str) -> "VLMConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
