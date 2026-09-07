"""Deterministic multi-segment utilities for text-aligned UCF features."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
PROMPT_TEMPLATES = (
    "a video of {label}",
    "a person performing {label}",
    "the human action {label}",
)


def humanize_ucf_class_name(name: str) -> str:
    """Convert UCF's CamelCase labels into a stable natural-language phrase."""
    value = re.sub(r"[_-]+", " ", str(name).strip())
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", value)
    return " ".join(value.lower().split())


def class_prompt_groups(
        class_names: Sequence[str],
        templates: Sequence[str] = PROMPT_TEMPLATES) -> Tuple[Tuple[str, ...], ...]:
    if not templates or any("{label}" not in template for template in templates):
        raise ValueError("Every prompt template must contain {label}")
    return tuple(tuple(
        template.format(label=humanize_ucf_class_name(name))
        for template in templates
    ) for name in class_names)


def segment_frame_indices(frame_count: int, num_segments: int = 3,
                          num_frames: int = 8) -> np.ndarray:
    """Uniformly sample frames independently inside equal temporal partitions."""
    if frame_count <= 0 or num_segments <= 0 or num_frames <= 0:
        raise ValueError("frame_count, num_segments, and num_frames must be positive")
    edges = np.linspace(0.0, float(frame_count), num_segments + 1)
    result = []
    for segment in range(num_segments):
        start = min(frame_count - 1, int(np.floor(edges[segment])))
        stop = min(frame_count, max(start + 1, int(np.floor(edges[segment + 1]))))
        result.append(np.linspace(
            start, stop - 1, num_frames, dtype=np.int64))
    return np.stack(result)


def decode_video_segments(path: Path, num_segments: int = 3,
                          num_frames: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Decode RGB frames as ``(segments, frames, height, width, channels)``."""
    try:
        import decord
    except ImportError as error:
        raise RuntimeError(
            "decord is required for LanguageBind video extraction") from error
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    indices = segment_frame_indices(len(reader), num_segments, num_frames)
    frames = reader.get_batch(indices.reshape(-1))
    if hasattr(frames, "asnumpy"):
        frames = frames.asnumpy()
    else:
        frames = np.asarray(frames)
    expected = num_segments * num_frames
    if frames.ndim != 4 or frames.shape[0] != expected or frames.shape[-1] != 3:
        raise ValueError(f"Unexpected decoded video shape for {path}: {frames.shape}")
    return frames.reshape(num_segments, num_frames, *frames.shape[1:]), indices


def preprocess_video_frames(frames: np.ndarray, size: int = 224) -> torch.Tensor:
    """Apply deterministic CLIP resize, center crop, and normalization."""
    values = torch.as_tensor(frames)
    if values.ndim == 4:
        values = values.unsqueeze(0)
    if values.ndim != 5 or values.shape[-1] != 3:
        raise ValueError(
            "frames must have shape (segments, frames, height, width, 3)")
    segments, frame_count, height, width, _ = values.shape
    values = values.permute(0, 1, 4, 2, 3).reshape(
        segments * frame_count, 3, height, width).float().div_(255.0)
    scale = float(size) / min(height, width)
    resized_height = max(size, int(round(height * scale)))
    resized_width = max(size, int(round(width * scale)))
    values = F.interpolate(
        values, size=(resized_height, resized_width), mode="bicubic",
        align_corners=False, antialias=True)
    top = (resized_height - size) // 2
    left = (resized_width - size) // 2
    values = values[:, :, top:top + size, left:left + size]
    mean = values.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = values.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    values = (values - mean) / std
    return values.reshape(segments, frame_count, 3, size, size).permute(0, 2, 1, 3, 4)


def ensemble_prompt_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Average normalized prompt embeddings and renormalize per class."""
    if embeddings.ndim != 3:
        raise ValueError("prompt embeddings must have shape (classes, prompts, dim)")
    return F.normalize(F.normalize(embeddings, dim=-1).mean(dim=1), dim=-1)


def temporal_class_logits(
        segment_embeddings: torch.Tensor, text_embeddings: torch.Tensor,
        valid_mask: torch.Tensor = None) -> torch.Tensor:
    """Mean class evidence over genuine temporal segments."""
    if segment_embeddings.ndim != 3 or text_embeddings.ndim != 2:
        raise ValueError(
            "segment embeddings must be (N,S,D) and text embeddings (C,D)")
    if segment_embeddings.shape[-1] != text_embeddings.shape[-1]:
        raise ValueError("segment and text embedding dimensions must match")
    segments = F.normalize(segment_embeddings, dim=-1)
    text = F.normalize(text_embeddings, dim=-1)
    logits = torch.einsum("nsd,cd->nsc", segments, text)
    if valid_mask is None:
        return logits.mean(dim=1)
    if valid_mask.shape != segment_embeddings.shape[:2]:
        raise ValueError("valid_mask must have shape (N,S)")
    weights = valid_mask.to(logits).unsqueeze(-1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    return (logits * weights).sum(dim=1) / counts
