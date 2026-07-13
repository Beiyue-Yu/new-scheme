import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.model_improvements import EmbeddingNet, Transformer


class IFNode(nn.Module):
    """Minimal hard-reset IF neuron used by the original vector MSTR branch."""

    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = float(threshold)
        self.v = None

    def reset(self):
        self.v = None

    def forward(self, x):
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        self.v = self.v + x
        hard = (self.v >= self.threshold).to(x.dtype)
        soft = torch.sigmoid(4.0 * (self.v - self.threshold))
        spike = hard.detach() + soft - soft.detach()
        self.v = self.v * (1.0 - hard)
        return spike


class SNNBranch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            IFNode(),
            nn.Linear(hidden_size, output_size),
            IFNode(),
        )

    def reset(self):
        for module in self.fc:
            if hasattr(module, "reset"):
                module.reset()

    def forward(self, x):
        return self.fc(x)


class VectorTRL(nn.Module):
    """Low-rank TRL specialized for pre-extracted vector features.

    With singleton spatial modes, the four-factor Tucker regression used by
    MSTR is algebraically just a low-rank linear map. Keeping the singleton
    factors makes gradients a product of several small values and caused all
    TRL weights to underflow. This two-factor form is the stable equivalent.
    """

    def __init__(self, input_size, output_size, rank):
        super().__init__()
        rank = min(int(rank), input_size, output_size)
        self.input_factor = nn.Linear(input_size, rank, bias=False)
        self.output_factor = nn.Linear(rank, output_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_size))
        nn.init.orthogonal_(self.input_factor.weight)
        nn.init.xavier_uniform_(self.output_factor.weight)

    def forward(self, x):
        return self.output_factor(self.input_factor(x)) + self.bias


class ResidualMSTR(nn.Module):
    """MSTR backbone with bounded, normalized vector-TRL residuals."""

    feature_dim = 300

    def __init__(self, params_model, input_size_audio, input_size_video):
        super().__init__()
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
        self.T = params_model.get("snn_T", 10)
        self.trl_gate_scale = float(params_model.get("trl_gate_scale", 0.25))

        self.pos_emb1D = nn.Parameter(torch.randn(2, self.feature_dim))
        self.pos_emb1D_t = nn.Parameter(torch.randn(2, self.feature_dim))

        self.A_enc = EmbeddingNet(
            input_size_audio, self.feature_dim, self.r_enc, True,
            self.momentum, self.hidden_size_encoder)
        self.V_enc = EmbeddingNet(
            input_size_video, self.feature_dim, self.r_enc, True,
            self.momentum, self.hidden_size_encoder)
        self.cross_attention = Transformer(
            self.feature_dim, self.depth_transformer, 3, 100, 64,
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

        trl_rank = params_model.get("vector_trl_rank", 64)
        self.vector_trl_a = VectorTRL(input_size_audio, self.feature_dim, trl_rank)
        self.vector_trl_v = VectorTRL(input_size_video, self.feature_dim, trl_rank)
        self.trl_norm_a = nn.LayerNorm(self.feature_dim, elementwise_affine=False)
        self.trl_norm_v = nn.LayerNorm(self.feature_dim, elementwise_affine=False)
        # Zero starts exactly at the proven MSTR backbone. The bounded gates can
        # admit TRL information only when validation-driven training supports it.
        self.trl_gate_a = nn.Parameter(torch.tensor(0.0))
        self.trl_gate_v = nn.Parameter(torch.tensor(0.0))

        self.criterion_reg = nn.MSELoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)

        trl_parameters = (
            list(self.vector_trl_a.parameters()) +
            list(self.vector_trl_v.parameters()) +
            [self.trl_gate_a, self.trl_gate_v]
        )
        trl_ids = {id(parameter) for parameter in trl_parameters}
        backbone_parameters = [
            parameter for parameter in self.parameters()
            if id(parameter) not in trl_ids
        ]
        backbone_lr_scale = float(params_model.get("backbone_lr_scale", 1.0))
        self.freeze_backbone = backbone_lr_scale == 0.0
        if self.freeze_backbone:
            for parameter in backbone_parameters:
                parameter.requires_grad_(False)
        optimizer_groups = [{"params": trl_parameters, "weight_decay": 0.0}]
        if not self.freeze_backbone:
            optimizer_groups.insert(0, {
                "params": backbone_parameters,
                "weight_decay": 1e-5,
                "lr": params_model["lr"] * backbone_lr_scale,
            })
        self.optimizer_gen = optim.Adam(
            optimizer_groups, lr=params_model["lr"], foreach=False)
        self.scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_gen, "max", patience=3, verbose=True)

    @property
    def trl_gates(self):
        return (
            self.trl_gate_scale * torch.tanh(self.trl_gate_a),
            self.trl_gate_scale * torch.tanh(self.trl_gate_v),
        )

    def optimize_scheduler(self, value):
        self.scheduler_gen.step(value)

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "freeze_backbone", False):
            for module in (
                self.A_enc, self.V_enc, self.cross_attention, self.W_proj,
                self.D, self.SNNbranchaudio, self.SNNbranchvideo,
                self.A_proj, self.V_proj, self.A_rec, self.V_rec,
            ):
                module.eval()
        return self

    def _encode_spikes(self, branch, x):
        branch.reset()
        spike_sum = 0.0
        try:
            for _ in range(self.T):
                spike_sum = spike_sum + branch(x)
            return spike_sum / self.T
        finally:
            branch.reset()

    def _semantic_features(self, audio, video):
        phi_a = self.A_enc(audio)
        phi_v = self.V_enc(video)
        spike_a = self._encode_spikes(self.SNNbranchaudio, audio)
        spike_v = self._encode_spikes(self.SNNbranchvideo, video)

        audio_tokens = torch.stack((
            phi_a + self.pos_emb1D[0],
            phi_a * F.softmax(spike_a, dim=1) + self.pos_emb1D[0],
        ), dim=1)
        video_tokens = torch.stack((
            phi_v + self.pos_emb1D[1],
            phi_v * F.softmax(spike_v, dim=1) + self.pos_emb1D[1],
        ), dim=1)
        return (
            self.cross_attention(audio_tokens)[:, 0],
            self.cross_attention(video_tokens)[:, 1],
        )

    def _spatial_features(self, audio, video):
        gate_a, gate_v = self.trl_gates
        trl_a = gate_a * self.trl_norm_a(self.vector_trl_a(audio))
        trl_v = gate_v * self.trl_norm_v(self.vector_trl_v(video))
        tokens = torch.stack((
            trl_a + self.pos_emb1D_t[0],
            trl_v + self.pos_emb1D_t[1],
        ), dim=1)
        attended = self.cross_attention(tokens)
        return trl_a + attended[:, 0], trl_v + attended[:, 1]

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

    def _encode_av(self, audio, video):
        phi_a, phi_v = self._semantic_features(audio, video)
        spatial_a, spatial_v = self._spatial_features(audio, video)
        fused_a, fused_v = self._fuse(phi_a, phi_v, spatial_a, spatial_v)
        return phi_a, phi_v, self.A_proj(fused_a), self.V_proj(fused_v)

    def forward(self, audio, video, negative_audio, negative_video,
                word_embedding, negative_word_embedding):
        self.phi_a, self.phi_v, self.theta_a, self.theta_v = self._encode_av(
            audio, video)
        _, _, self.theta_a_neg, self.theta_v_neg = self._encode_av(
            negative_audio, negative_video)
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
                    self.triplet_loss(self.theta_a, self.theta_w, self.theta_a_neg) +
                    self.triplet_loss(self.theta_v, self.theta_w, self.theta_v_neg)) +
                self.second_additional_triplet * (
                    self.triplet_loss(self.theta_w, self.theta_a, self.theta_w_neg) +
                    self.triplet_loss(self.theta_w, self.theta_v, self.theta_w_neg))
            )
        l_reg = self.theta_w.new_tensor(0.0)
        if self.reg_loss:
            l_reg = (
                self.criterion_reg(self.phi_v_rec, self.phi_v) +
                self.criterion_reg(self.phi_a_rec, self.phi_a) +
                self.criterion_reg(self.theta_v, self.theta_w) +
                self.criterion_reg(self.theta_a, self.theta_w)
            )
        l_rec = (
            self.criterion_reg(self.w, self.rho_v) +
            self.criterion_reg(self.w, self.rho_a) +
            self.criterion_reg(self.w, self.rho_w)
        )
        l_cross = (
            self.triplet_loss(self.rho_w, self.rho_v, self.rho_v_neg) +
            self.triplet_loss(self.rho_w, self.rho_a, self.rho_a_neg)
        )
        l_word = (
            self.triplet_loss(self.theta_w, self.theta_v, self.theta_v_neg) +
            self.triplet_loss(self.theta_w, self.theta_a, self.theta_a_neg) +
            self.triplet_loss(self.theta_a, self.theta_w, self.theta_w_neg) +
            self.triplet_loss(self.theta_v, self.theta_w, self.theta_w_neg)
        )
        loss = l_rec + l_cross + l_word + l_additional + l_reg
        if optimize:
            self.optimizer_gen.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            self.optimizer_gen.step()
        details = {
            "gen": loss, "reconstruction": l_rec, "cross": l_cross,
            "word": l_word, "additional": l_additional, "reg": l_reg,
        }
        return loss, details

    def optimize_params(self, audio, video, cls_numeric, cls_embedding,
                        audio_negative, video_negative, negative_cls_embedding,
                        optimize=False):
        self.forward(audio, video, audio_negative, video_negative,
                     cls_embedding, negative_cls_embedding)
        return self.backward(optimize)

    def get_embeddings(self, audio, video, embedding):
        _, _, theta_a, theta_v = self._encode_av(audio, video)
        return theta_a, theta_v, self.W_proj(embedding)
