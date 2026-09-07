import json
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from build_languagebind_mstr_features import _convert_split, _load_caches
from src.model_improvements import MSTR
from src.utils_improvements import get_model_params


def _write_cache(path: Path):
    metadata = {
        "selected_class_ids": [2, 7],
        "model_sha256": "test-model",
        "encoded_videos": 2,
    }
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata)),
        video_names=np.asarray(["clip-a", "clip-b"]),
        video_embeddings=np.asarray([
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ], dtype=np.float32),
        text_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


def _write_processed(path: Path, urls=("clip-b", "clip-a"),
                     text_targets=None):
    targets = torch.tensor([7, 2])
    if text_targets is None:
        text_targets = torch.arange(8)
    text_targets = torch.as_tensor(text_targets)
    payload = {
        "audio": {
            "data": torch.ones(2, 3), "target": targets,
            "url": np.asarray(urls),
        },
        "video": {
            "data": torch.zeros(2, 3), "target": targets,
            "url": np.asarray(urls),
        },
        "text": {
            "data": torch.zeros(8, 3), "target": text_targets,
            "url": np.asarray([]),
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


class LanguageBindMSTRFeatureTests(unittest.TestCase):
    def test_builder_joins_languagebind_features_by_stable_url(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "cache.npz"
            source = root / "source.pkl"
            destination = root / "destination.pkl"
            _write_cache(cache_path)
            _write_processed(source)
            videos, texts, provenance = _load_caches([cache_path])
            _convert_split(source, destination, videos, texts, provenance)
            with destination.open("rb") as handle:
                result = pickle.load(handle)
            torch.testing.assert_close(
                result["video"]["data"],
                torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
            torch.testing.assert_close(
                result["text"]["data"][2], torch.tensor([1.0, 0.0]))
            torch.testing.assert_close(
                result["text"]["data"][7], torch.tensor([0.0, 1.0]))
            self.assertEqual(result["audio"]["data"].shape, (2, 3))

    def test_builder_writes_text_by_class_id_when_source_rows_are_reordered(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "cache.npz"
            source = root / "source.pkl"
            destination = root / "destination.pkl"
            _write_cache(cache_path)
            # UCF's text target list moves class 2 to the final active row.
            _write_processed(source, text_targets=[0, 1, 7, 3, 4, 5, 6, 2])
            videos, texts, provenance = _load_caches([cache_path])
            _convert_split(source, destination, videos, texts, provenance)
            with destination.open("rb") as handle:
                result = pickle.load(handle)
            torch.testing.assert_close(
                result["text"]["data"][2], torch.tensor([1.0, 0.0]))
            torch.testing.assert_close(
                result["text"]["data"][7], torch.tensor([0.0, 1.0]))

    def test_builder_rejects_incomplete_video_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "cache.npz"
            source = root / "source.pkl"
            _write_cache(cache_path)
            _write_processed(source, urls=("clip-b", "missing"))
            videos, texts, provenance = _load_caches([cache_path])
            with self.assertRaisesRegex(
                    ValueError, "absent from LanguageBind caches"):
                _convert_split(
                    source, root / "destination.pkl", videos, texts, provenance)

    def test_stft_mstr_accepts_languagebind_text_dimension(self):
        params = get_model_params(
            1e-3, 1, 1, True, True, 0.1, 0.1, 0.1,
            8, 8, 1, 0.1, stft_dim=8, trl_rank=4,
            lkc_n_heads=2, tucker_rank=4, text_embedding_size=12,
            use_glp=False)
        model = MSTR(params, input_size_audio=8, input_size_video=10).eval()
        projected = model.W_proj(torch.randn(2, 12))
        reconstructed = model.D(projected)
        self.assertEqual(projected.shape, (2, 64))
        self.assertEqual(reconstructed.shape, (2, 12))


if __name__ == "__main__":
    unittest.main()
