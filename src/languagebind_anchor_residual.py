"""LanguageBind semantic-anchor residual model.

The model deliberately keeps the frozen LanguageBind video/text geometry in
its native shared space.  Only a zero-initialized, low-capacity video residual
and a weak audio adapter are trained.  It implements the small API surface
used by the MSTR training/evaluation loop without changing MSTR checkpoints.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class LanguageBindAnchorResidual(nn.Module):
    """Frozen LanguageBind anchor plus validation-gated trainable residuals."""

    def __init__(self, params_model, input_size_audio, input_size_video):
        super().__init__()
        self.text_embedding_size = int(
            params_model.get("text_embedding_size", input_size_video))
        if self.text_embedding_size != int(input_size_video):
            raise ValueError(
                "LanguageBind anchor requires equal video/text dimensions; "
                f"video={input_size_video}, text={self.text_embedding_size}")
        if self.text_embedding_size <= 0:
            raise ValueError("text_embedding_size must be positive")

        hidden = int(params_model.get("anchor_hidden_size", 128))
        if hidden <= 0:
            raise ValueError("anchor_hidden_size must be positive")
        self.margin = float(params_model.get("anchor_margin", 0.05))
        self.residual_scale = float(params_model.get("anchor_residual_scale", 0.10))
        self.audio_scale = float(params_model.get("anchor_audio_scale", 0.05))
        self.residual_weight = float(
            params_model.get("anchor_residual_weight", 0.01))
        if self.margin < 0.0 or self.residual_scale < 0.0:
            raise ValueError("anchor margin and residual scale must be non-negative")
        if self.audio_scale < 0.0 or self.residual_weight < 0.0:
            raise ValueError("anchor audio scale and residual weight must be non-negative")

        self.video_residual = nn.Sequential(
            nn.Linear(input_size_video, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, input_size_video),
        )
        # Start exactly at the frozen LanguageBind video anchor.
        nn.init.zeros_(self.video_residual[-1].weight)
        nn.init.zeros_(self.video_residual[-1].bias)

        self.audio_adapter = nn.Sequential(
            nn.Linear(input_size_audio, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, input_size_video),
        )
        self.lr = float(params_model.get("lr", 1e-3))
        self.optimizer_gen = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_gen, mode="max", patience=3)
        self.batch_labels = None
        self._last_loss = None

    @staticmethod
    def _normalize(value):
        return F.normalize(value.float(), dim=1, eps=1e-8)

    def _video_anchor(self, video):
        anchor = self._normalize(video)
        residual = self.video_residual(anchor)
        corrected = self._normalize(anchor + self.residual_scale * residual)
        return corrected, residual

    def _audio_projection(self, audio):
        # A small output scale keeps weak audio from overwhelming the strong
        # frozen video branch when the evaluator sums modality distances.
        projected = self._normalize(self.audio_adapter(audio))
        return self.audio_scale * projected

    def _text_anchor(self, text):
        return self._normalize(text)

    def forward(self, audio, video, negative_audio, negative_video,
                word_embedding, negative_word_embedding):
        audio_p = self._audio_projection(audio)
        video_p, _ = self._video_anchor(video)
        text_p = self._text_anchor(word_embedding)
        audio_n = self._audio_projection(negative_audio)
        video_n, _ = self._video_anchor(negative_video)
        text_n = self._text_anchor(negative_word_embedding)
        # Match the legacy tuple consumed by evaluate_dataset.  The decoder
        # views are intentionally absent because this route has no decoder.
        return (None, audio_p, video_p, text_p, audio_n, video_n, text_n,
                None, None, None, None, None)

    @staticmethod
    def _ranking_loss(embedding, positive_text, negative_text, margin):
        # Compare directions, not norms, so the fixed audio scale cannot make
        # its auxiliary objective numerically dominant.
        embedding = F.normalize(embedding, dim=1)
        positive_text = F.normalize(positive_text, dim=1)
        negative_text = F.normalize(negative_text, dim=1)
        positive = (embedding * positive_text).sum(dim=1)
        negative = (embedding * negative_text).sum(dim=1)
        return F.relu(margin - positive + negative).mean()

    def optimize_params(self, audio, video, cls_numeric, cls_embedding,
                        audio_negative, video_negative,
                        negative_cls_embedding, optimize=False,
                        teacher_embeddings=None, teacher_mask=None,
                        teacher_weight=0.0):
        del cls_numeric, teacher_embeddings, teacher_mask, teacher_weight
        self.batch_labels = None
        audio_p = self._audio_projection(audio)
        video_p, video_residual = self._video_anchor(video)
        text_p = self._text_anchor(cls_embedding)
        audio_n = self._audio_projection(audio_negative)
        video_n, _ = self._video_anchor(video_negative)
        text_n = self._text_anchor(negative_cls_embedding)

        video_loss = self._ranking_loss(
            video_p, text_p, text_n, self.margin)
        audio_loss = self._ranking_loss(
            audio_p, text_p, text_n, self.margin)
        residual_penalty = video_residual.square().mean()
        loss = video_loss + 0.15 * audio_loss + self.residual_weight * residual_penalty
        if optimize:
            self.optimizer_gen.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            self.optimizer_gen.step()
        self._last_loss = loss.detach()
        return loss, {
            "gen": loss.detach(),
            "anchor_video": video_loss.detach(),
            "anchor_audio": audio_loss.detach(),
            "anchor_residual": residual_penalty.detach(),
        }

    def optimize_scheduler(self, value):
        self.scheduler_gen.step(value)

    def get_embeddings(self, audio, video, embedding):
        audio_p = self._audio_projection(audio)
        video_p, _ = self._video_anchor(video)
        text_p = self._text_anchor(embedding)
        return audio_p, video_p, text_p

    def get_runtime_diagnostics(self):
        return {
            "anchor_residual_norm": (
                self.video_residual[-1].weight.detach().norm())
        }
