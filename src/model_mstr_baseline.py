import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.model_improvements import EmbeddingNet, Transformer, TRL
from src.model_residual import SNNBranch


class MSTRBaseline(nn.Module):
    """Stable reproduction of the released MSTR architecture.

    ``mstr_released`` preserves the released transformer's 3 x 100 layout and
    its mixed video/audio SNN recurrence. ``mstr_paper`` fixes the modality
    recurrence, resets every sample independently, and uses the paper's 8 x 64
    CMF transformer. Both modes remove the released code's fixed batch reshape
    and use the pure-PyTorch Tucker regression implementation.
    """

    feature_dim = 300

    def __init__(self, params_model, input_size_audio, input_size_video):
        super().__init__()
        self.variant = params_model.get("fusion_mode", "mstr_paper")
        if self.variant not in {"mstr_released", "mstr_paper"}:
            raise ValueError(f"Unsupported MSTR baseline variant: {self.variant}")
        self.release_compat = self.variant == "mstr_released"
        if self.release_compat and input_size_audio != input_size_video:
            raise ValueError(
                "mstr_released reproduces the released mixed audio/video SNN "
                "recurrence and therefore requires equal input dimensions")

        self.dim_out = params_model["dim_out"]
        self.hidden_size_encoder = params_model["encoder_hidden_size"]
        self.hidden_size_decoder = params_model["decoder_hidden_size"]
        self.r_enc = params_model["dropout_encoder"]
        self.r_proj = params_model["dropout_decoder"]
        self.r_dec = params_model["additional_dropout"]
        self.depth_transformer = params_model["depth_transformer"]
        self.additional_triplets_loss = params_model["additional_triplets_loss"]
        self.reg_loss = params_model["reg_loss"]
        self.momentum = params_model["momentum"]
        self.first_additional_triplet = params_model["first_additional_triplet"]
        self.second_additional_triplet = params_model["second_additional_triplet"]
        self.T = 10 if self.release_compat else params_model.get("snn_T", 10)

        self.A_enc = EmbeddingNet(
            input_size_audio, self.feature_dim, self.r_enc, True,
            self.momentum, self.hidden_size_encoder)
        self.V_enc = EmbeddingNet(
            input_size_video, self.feature_dim, self.r_enc, True,
            self.momentum, self.hidden_size_encoder)

        trl_rank = int(params_model.get("trl_rank", 400))
        self.trl_a = TRL(
            ranks=(trl_rank, 1, 1, self.feature_dim),
            input_size=(1, input_size_audio, 1, 1),
            output_size=(1, self.feature_dim))
        self.trl_v = TRL(
            ranks=(trl_rank, 1, 1, self.feature_dim),
            input_size=(1, input_size_video, 1, 1),
            output_size=(1, self.feature_dim))

        heads, dim_head = ((3, 100) if self.release_compat else (8, 64))
        self.cross_attention = Transformer(
            self.feature_dim, self.depth_transformer, heads, dim_head, 64,
            dropout=self.r_enc)

        self.W_proj = EmbeddingNet(
            self.feature_dim, self.dim_out, self.r_dec, True, self.momentum)
        self.D = EmbeddingNet(
            self.dim_out, self.feature_dim, self.r_dec, True, self.momentum)
        self.SNNbranchaudio = SNNBranch(
            input_size_audio, self.hidden_size_encoder, self.feature_dim)
        self.SNNbranchvideo = SNNBranch(
            input_size_video, self.hidden_size_encoder, self.feature_dim)
        self.A_proj = EmbeddingNet(
            self.feature_dim, self.dim_out, self.r_proj, True,
            self.momentum, self.hidden_size_decoder)
        self.V_proj = EmbeddingNet(
            self.feature_dim, self.dim_out, self.r_proj, True,
            self.momentum, self.hidden_size_decoder)
        self.A_rec = EmbeddingNet(
            self.dim_out, self.feature_dim, self.r_dec, True, self.momentum)
        self.V_rec = EmbeddingNet(
            self.dim_out, self.feature_dim, self.r_dec, True, self.momentum)

        self.pos_emb1D = nn.Parameter(torch.randn(2, self.feature_dim))
        self.pos_emb1D_t = nn.Parameter(torch.randn(2, self.feature_dim))
        self.criterion_reg = nn.MSELoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
        self.optimizer_gen = optim.Adam(
            self.parameters(), lr=params_model["lr"], weight_decay=1e-5,
            foreach=False)
        self.scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_gen, "max", patience=3, verbose=True)

    def optimize_scheduler(self, value):
        self.scheduler_gen.step(value)

    def _reset_snn(self):
        self.SNNbranchaudio.reset()
        self.SNNbranchvideo.reset()

    def _mean_spikes(self, branch, features):
        branch.reset()
        try:
            output = 0.0
            for _ in range(self.T):
                output = output + branch(features)
            return output / self.T
        finally:
            branch.reset()

    def _released_spikes(self, audio, video, negative_audio=None,
                         negative_video=None):
        """Reproduce the released mixed recurrence without fixed batch shapes."""
        self._reset_snn()
        try:
            audio_spikes = sum(
                (self.SNNbranchaudio(audio) for _ in range(self.T)),
                audio.new_zeros(audio.shape[0], self.feature_dim)) / self.T
            video_spikes = self.SNNbranchvideo(video)
            for _ in range(1, self.T):
                video_spikes = video_spikes + self.SNNbranchaudio(video)
            video_spikes = video_spikes / self.T
            if negative_audio is None:
                return audio_spikes, video_spikes

            negative_audio_spikes = 0.0
            for _ in range(self.T):
                negative_audio_spikes = (
                    negative_audio_spikes + self.SNNbranchaudio(negative_audio))
            negative_audio_spikes = negative_audio_spikes / self.T
            negative_video_spikes = self.SNNbranchvideo(negative_video)
            for _ in range(1, self.T):
                negative_video_spikes = (
                    negative_video_spikes + self.SNNbranchaudio(negative_video))
            return (audio_spikes, video_spikes, negative_audio_spikes,
                    negative_video_spikes / self.T)
        finally:
            self._reset_snn()

    def _attend_temporal(self, semantic, spikes, modality_index):
        tokens = torch.stack((
            semantic + self.pos_emb1D[modality_index],
            semantic * F.softmax(spikes, dim=1) +
            self.pos_emb1D[modality_index],
        ), dim=1)
        return self.cross_attention(tokens)[:, modality_index]

    def _semantic_features(self, audio, video):
        phi_a = self.A_enc(audio)
        phi_v = self.V_enc(video)
        if self.release_compat:
            spike_a, spike_v = self._released_spikes(audio, video)
        else:
            spike_a = self._mean_spikes(self.SNNbranchaudio, audio)
            spike_v = self._mean_spikes(self.SNNbranchvideo, video)
        return (
            self._attend_temporal(phi_a, spike_a, 0),
            self._attend_temporal(phi_v, spike_v, 1),
        )

    @staticmethod
    def _as_spatial_tensor(features):
        return features.unsqueeze(-1).unsqueeze(-1)

    def _spatial_features(self, audio, video):
        spatial_a = self.trl_a(self._as_spatial_tensor(audio))
        spatial_v = self.trl_v(self._as_spatial_tensor(video))
        tokens = torch.stack((
            spatial_a + self.pos_emb1D_t[0],
            spatial_v + self.pos_emb1D_t[1],
        ), dim=1)
        attended = self.cross_attention(tokens)
        return spatial_a + attended[:, 0], spatial_v + attended[:, 1]

    def _fuse(self, phi_a, phi_v, spatial_a, spatial_v):
        semantic = self.cross_attention(torch.stack((
            phi_a + self.pos_emb1D[0],
            phi_v + self.pos_emb1D[1],
        ), dim=1))
        audio_base = phi_a + semantic[:, 0]
        video_base = phi_v + semantic[:, 1]
        audio = self.cross_attention(torch.stack((
            audio_base + self.pos_emb1D[0],
            spatial_v + self.pos_emb1D_t[0],
        ), dim=1))[:, 0]
        video = self.cross_attention(torch.stack((
            video_base + self.pos_emb1D[1],
            spatial_a + self.pos_emb1D_t[1],
        ), dim=1))[:, 0]
        return audio, video

    def _project(self, phi_a, phi_v, audio, video):
        spatial_a, spatial_v = self._spatial_features(audio, video)
        fused_a, fused_v = self._fuse(phi_a, phi_v, spatial_a, spatial_v)
        return self.A_proj(fused_a), self.V_proj(fused_v)

    def forward(self, audio, video, negative_audio, negative_video,
                word_embedding, negative_word_embedding):
        if self.release_compat:
            semantic_a = self.A_enc(audio)
            semantic_v = self.V_enc(video)
            semantic_a_neg = self.A_enc(negative_audio)
            semantic_v_neg = self.V_enc(negative_video)
            spikes = self._released_spikes(
                audio, video, negative_audio, negative_video)
            self.phi_a = self._attend_temporal(semantic_a, spikes[0], 0)
            self.phi_v = self._attend_temporal(semantic_v, spikes[1], 1)
            self.phi_a_neg = self._attend_temporal(
                semantic_a_neg, spikes[2], 0)
            self.phi_v_neg = self._attend_temporal(
                semantic_v_neg, spikes[3], 1)
        else:
            self.phi_a, self.phi_v = self._semantic_features(audio, video)
            self.phi_a_neg, self.phi_v_neg = self._semantic_features(
                negative_audio, negative_video)

        self.theta_a, self.theta_v = self._project(
            self.phi_a, self.phi_v, audio, video)
        self.theta_a_neg, self.theta_v_neg = self._project(
            self.phi_a_neg, self.phi_v_neg, negative_audio, negative_video)
        self.w = word_embedding
        self.w_neg = negative_word_embedding
        self.theta_w = self.W_proj(word_embedding)
        self.theta_w_neg = self.W_proj(negative_word_embedding)
        self.phi_a_rec = self.A_rec(self.theta_a)
        self.phi_v_rec = self.V_rec(self.theta_v)
        self.rho_a = self.D(self.theta_a)
        self.rho_v = self.D(self.theta_v)
        self.rho_w = self.D(self.theta_w)
        self.rho_a_neg = self.D(self.theta_a_neg)
        self.rho_v_neg = self.D(self.theta_v_neg)

    def backward(self, optimize):
        l_additional = self.theta_w.new_tensor(0.0)
        if self.additional_triplets_loss:
            l_additional = (
                self.first_additional_triplet * (
                    self.triplet_loss(
                        self.theta_a, self.theta_w, self.theta_a_neg) +
                    self.triplet_loss(
                        self.theta_v, self.theta_w, self.theta_v_neg)) +
                self.second_additional_triplet * (
                    self.triplet_loss(
                        self.theta_w, self.theta_a, self.theta_w_neg) +
                    self.triplet_loss(
                        self.theta_w, self.theta_v, self.theta_w_neg)))

        l_reg = self.theta_w.new_tensor(0.0)
        if self.reg_loss:
            l_reg = (
                self.criterion_reg(self.phi_v_rec, self.phi_v) +
                self.criterion_reg(self.phi_a_rec, self.phi_a) +
                self.criterion_reg(self.theta_v, self.theta_w) +
                self.criterion_reg(self.theta_a, self.theta_w))

        l_rec = (
            self.criterion_reg(self.w, self.rho_v) +
            self.criterion_reg(self.w, self.rho_a) +
            self.criterion_reg(self.w, self.rho_w))
        l_cross = (
            self.triplet_loss(self.rho_w, self.rho_v, self.rho_v_neg) +
            self.triplet_loss(self.rho_w, self.rho_a, self.rho_a_neg))
        l_word = (
            self.triplet_loss(self.theta_w, self.theta_v, self.theta_v_neg) +
            self.triplet_loss(self.theta_w, self.theta_a, self.theta_a_neg) +
            self.triplet_loss(self.theta_a, self.theta_w, self.theta_w_neg) +
            self.triplet_loss(self.theta_v, self.theta_w, self.theta_w_neg))
        loss = l_rec + l_cross + l_word + l_additional + l_reg
        if optimize:
            self.optimizer_gen.zero_grad()
            loss.backward()
            self.optimizer_gen.step()
        return loss, {
            "gen": loss, "reconstruction": l_rec, "cross": l_cross,
            "word": l_word, "additional": l_additional, "reg": l_reg,
        }

    def optimize_params(self, audio, video, cls_numeric, cls_embedding,
                        audio_negative, video_negative, negative_cls_embedding,
                        optimize=False):
        self.forward(audio, video, audio_negative, video_negative,
                     cls_embedding, negative_cls_embedding)
        return self.backward(optimize)

    def get_embeddings(self, audio, video, embedding):
        phi_a, phi_v = self._semantic_features(audio, video)
        theta_a, theta_v = self._project(phi_a, phi_v, audio, video)
        return theta_a, theta_v, self.W_proj(embedding)
