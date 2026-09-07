#!/usr/bin/env python3
"""Build MSTR-compatible processed UCF features from frozen LanguageBind caches.

The original SeLaVi audio observations are retained.  Video observations are
replaced with the normalized mean of LanguageBind's temporal video segments,
and the class text dictionary is replaced with the matching LanguageBind text
embeddings.  Samples are joined by the repository's stable ``url`` field so a
cache can never silently reorder the MSTR examples.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


SPLIT_FILES = {
    "train": "trainingmain_split.pkl",
    "val": "valmain_split.pkl",
    "train_val": "train_valmain_split.pkl",
    "test": "testmain_split.pkl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_source", type=Path, required=True)
    parser.add_argument("--processed_destination", type=Path, required=True)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=tuple(SPLIT_FILES), action="append")
    parser.add_argument(
        "--allow_missing", action="store_true",
        help="Write zero placeholders for missing samples/classes; never use "
             "this mode for a validation or test report")
    return parser.parse_args()


def _load_caches(paths: Sequence[Path]):
    videos: Dict[str, np.ndarray] = {}
    texts: Dict[int, np.ndarray] = {}
    provenance = []
    for path in paths:
        with np.load(path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata_json"]))
            names = cache["video_names"].astype(str)
            embeddings = cache["video_embeddings"].astype(np.float32)
            if embeddings.ndim != 3 or embeddings.shape[1] < 1:
                raise ValueError(f"Unexpected LanguageBind video shape in {path}")
            for name, embedding in zip(names, embeddings):
                value = embedding.mean(axis=0)
                value /= max(float(np.linalg.norm(value)), 1e-12)
                previous = videos.get(str(name))
                if previous is not None and not np.allclose(previous, value, atol=2e-5):
                    raise ValueError(f"Conflicting LanguageBind embedding for {name}")
                videos[str(name)] = value.astype(np.float32)
            class_ids = np.asarray(metadata["selected_class_ids"], dtype=np.int64)
            text_embeddings = cache["text_embeddings"].astype(np.float32)
            if text_embeddings.shape[0] != class_ids.shape[0]:
                raise ValueError(f"Text ids and embeddings are misaligned in {path}")
            for class_id, embedding in zip(class_ids, text_embeddings):
                value = embedding / max(float(np.linalg.norm(embedding)), 1e-12)
                previous = texts.get(int(class_id))
                if previous is not None and not np.allclose(previous, value, atol=2e-5):
                    raise ValueError(f"Conflicting LanguageBind text for class {class_id}")
                texts[int(class_id)] = value.astype(np.float32)
            provenance.append({
                "path": str(path),
                "model_sha256": metadata.get("model_sha256"),
                "encoded_videos": int(metadata.get("encoded_videos", len(names))),
            })
    if not videos or not texts:
        raise ValueError("LanguageBind caches must contain video and text embeddings")
    return videos, texts, provenance


def _convert_split(source: Path, destination: Path, videos: Mapping[str, np.ndarray],
                   texts: Mapping[int, np.ndarray], provenance,
                   allow_missing: bool = False) -> dict:
    with source.open("rb") as handle:
        data = pickle.load(handle)
    urls = np.asarray(data["video"]["url"]).astype(str)
    missing = sorted(set(urls.tolist()) - set(videos))
    if missing and not allow_missing:
        raise ValueError(
            f"{source.name}: {len(missing)} samples are absent from LanguageBind "
            f"caches; first missing samples: {missing[:5]}")
    class_targets = torch.as_tensor(data["text"]["target"], dtype=torch.long)
    active_classes = set(torch.as_tensor(
        data["video"]["target"], dtype=torch.long).tolist())
    missing_classes = sorted(active_classes - set(texts))
    if missing_classes and not allow_missing:
        raise ValueError(
            f"{source.name}: missing LanguageBind text for class ids "
            f"{missing_classes}")
    video_dim = next(iter(videos.values())).shape[0]
    video = torch.from_numpy(np.stack([
        videos.get(name, np.zeros(video_dim, dtype=np.float32))
        for name in urls]))
    text_dim = next(iter(texts.values())).shape[0]
    # UCF's text target list is not guaranteed to be in class-id order (it
    # contains a historical reordered tail).  The dataset loader indexes text
    # rows directly by class id, so write by class id rather than enumerate
    # position.  Position-based assignment silently gives Unseen classes the
    # wrong prototype or a zero vector.
    text = torch.zeros((len(class_targets), text_dim), dtype=torch.float32)
    for class_id in class_targets.tolist():
        if class_id in texts:
            text[int(class_id)] = torch.from_numpy(texts[class_id])
    result = {
        "audio": data["audio"],
        "video": {
            "data": video,
            "target": data["video"]["target"],
            "url": data["video"]["url"],
        },
        "text": {
            "data": text,
            "target": data["text"]["target"],
            "url": data["text"].get("url", np.asarray([])),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(result, handle, pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "samples": int(len(urls)),
        "video_dim": int(video.shape[1]),
        "text_dim": int(text.shape[1]),
        "missing_video_samples": len(missing),
        "missing_text_classes": missing_classes,
    }


def main() -> None:
    args = parse_args()
    selected = args.split or list(SPLIT_FILES)
    videos, texts, provenance = _load_caches(args.cache)
    reports = []
    for split in selected:
        source = args.processed_source / SPLIT_FILES[split]
        destination = args.processed_destination / SPLIT_FILES[split]
        reports.append(_convert_split(
            source, destination, videos, texts, provenance,
            allow_missing=args.allow_missing))
    report = {
        "format_version": 1,
        "feature_route": "LanguageBind video + LanguageBind text + original audio",
        "video_samples_available": len(videos),
        "text_classes_available": len(texts),
        "cache_provenance": provenance,
        "splits": reports,
    }
    args.processed_destination.mkdir(parents=True, exist_ok=True)
    with (args.processed_destination / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    for item in reports:
        print(
            f"built={item['destination']} samples={item['samples']} "
            f"video_dim={item['video_dim']} text_dim={item['text_dim']}",
            flush=True)


if __name__ == "__main__":
    main()
