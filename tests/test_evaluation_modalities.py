import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.utils import evaluate_dataset_baseline


class _Dataset:
    all_class_ids = np.array([0, 1, 2, 3])
    seen_class_ids = np.array([0, 1])
    unseen_class_ids = np.array([2, 3])
    all_class_names = np.array(["zero", "one", "two", "three"])
    dataset_name = "synthetic"

    def __init__(self):
        self.targets = torch.arange(4)
        classes = torch.eye(4)
        self.all_data = {
            "audio": classes.clone(),
            "video": classes.roll(shifts=1, dims=0),
            "text": classes.clone(),
            "target": self.targets.clone(),
        }


class _IdentityModel:
    def eval(self):
        return self

    def get_embeddings(self, audio, video, text):
        assert not torch.is_grad_enabled()
        return audio, video, text


class EvaluationModalitiesTest(unittest.TestCase):
    def test_audio_video_and_both_are_evaluated_independently(self):
        result = evaluate_dataset_baseline(
            dataset=_Dataset(),
            model=_IdentityModel(),
            device="cpu",
            distance_fn="L2Loss",
            new_model_attention=True,
            args=SimpleNamespace(z_score_inputs=False, cjme=False),
        )

        self.assertEqual(result["audio"]["seen"], 1.0)
        self.assertEqual(result["audio"]["zsl"], 1.0)
        self.assertNotEqual(result["audio"]["seen"], result["video"]["seen"])
        self.assertIsNot(result["audio"], result["video"])
        self.assertIsNot(result["video"], result["both"])


if __name__ == "__main__":
    unittest.main()
